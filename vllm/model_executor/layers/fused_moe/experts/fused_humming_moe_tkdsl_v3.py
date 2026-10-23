"""v3 HummingIndexedExperts.apply: INT3 (group 128) routed experts, decode path m <= 64 in new TKDSL kernels.

Decode (m <= 64): routing prep (per-expert pick lists, a list of active (expert, 4-token pass) work units) ->
INT3 gate/up grouped GEMV -> clamped SwiGLU -> INT3 down grouped GEMV (top-k weight applied) -> bf16.  The GEMVs are
stream-K: every CTA takes one contiguous, equal range of 128-K stages over (pass, N tile, K group), so the grid ends
together whatever the routing.  m > 64 calls #54's apply (its tcgen05 prefill path) unchanged.
"""

import torch
from tkdsl import nvidia as tk  # noqa: F401

from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import *  # noqa: F401,F403
from vllm.model_executor.layers.fused_moe.experts import fused_humming_moe_tkdsl as _v2

K1, N1 = 5120, 4608
I2 = 2304
K2, N2 = 2304, 5120
TOPK = 6
CAP = 64          # max picks per expert on the decode path (m <= 64, ids distinct per token)
DECODE_MAX = 64

WPR1 = N1 * 3 // 2               # u32 words per 16-K row of w13
W1E = (K1 // 16) * WPR1          # words per expert of w13
WPR2 = N2 * 3 // 2
W2E = (K2 // 16) * WPR2

import os as _os
STAGES = int(_os.environ.get("V3_STAGES", "2"))
NCW = 8                          # consumer warps
THREADS = (NCW + 1) * 32
SEG = 768                        # u32 words per 16-K row of a stage (8 chunks x 96)
RPS = int(_os.environ.get("V3_RPS", "8"))  # 16-K weight rows per pipeline stage (8: one 128-K group, 4: half)
SPG = 8 // RPS                   # pipeline stages per 128-K group
REC = 68                         # u32 words per (token, 128-K group) record
XROW = 80                        # padded words per token row in the x stage
REC2 = 2 * XROW
TPP = 4                          # tokens per pass (two B columns per token)
CORR_A = -(1024.0 / 2 - 16 * 128.0 + 4)
CORR_B = -(128.0 + 4)
STAGE_X = 8 * XROW
STAGE_W = RPS * SEG
G1 = K1 // 128
G2 = K2 // 128
T1 = I2 // 256                   # gate/up tiles per expert
T2 = N2 // 512                   # down tiles per expert
SSTAGE = 256                     # u32 words of bf16 group scales per stage (512 scale columns)
SMEM = STAGES * (STAGE_W + STAGE_X + SSTAGE) * 4 + 256

PREP_THREADS = 512
CPS = int(_os.environ.get("V3_CPS", "3"))  # persistent CTAs per SM
MAXREG = int(_os.environ.get("V3_MAXREG", "0")) or None
ABL = _os.environ.get("V3_ABL", "")
PD = int(_os.environ.get("V3_PD", "0"))  # 128-K groups of weights prefetched into L2 ahead (measured: slower)
ZCTAS = int(_os.environ.get("V3_ZCTAS", "64"))
SPLIT1 = int(_os.environ.get("V3_SPLIT1", "8"))  # small chunks per gu item in the tail phase (G1=40 groups)
SPLIT2 = int(_os.environ.get("V3_SPLIT2", "6"))  # small chunks per dn item in the tail phase (G2=18 groups)
KS = int(_os.environ.get("V3_KS", "1"))          # tail-phase items per CTA
TMAP = int(_os.environ.get("V3_TMAP", "1"))
L2P = int(_os.environ.get("V3_L2P", "0")) or None  # TMA L2 promotion bytes (measured: no effect)
PDL = int(_os.environ.get("V3_PDL", "1"))
SMAP = int(_os.environ.get("V3_SMAP", "1"))
_WCACHE = tk.tma.CachePolicy.EVICT_FIRST if int(_os.environ.get("V3_EF", "1")) else tk.tma.CachePolicy.DEFAULT  # gu's gate|up scale rows of a stage in one TMA tensor-map copy  # programmatic dependent launch between the decode kernels


@tk.inline
def _write_rec(dst, rec, lane, p0, p1):
    """Lane holds k = 4*lane..4*lane+3 of one 128-K group as two bf16x2 words; writes the f16/bf16 record pair of
    the paired-code decode (type 0 at 0, type 1 at +XROW) and the f32 zero-point correction at word 0."""
    r = lane >> 2
    c4 = lane & 3
    isb = c4 >> 1
    a0, a1 = tk.ops.unpack(tk.ops.bitcast(p0, "bf16x2"))
    a2, a3 = tk.ops.unpack(tk.ops.bitcast(p1, "bf16x2"))
    s = (a0 + a1) + (a2 + a3)
    coef = tk.ops.select(isb > 0, CORR_B, CORR_A)
    tot = tk.warp.sum(s * coef)
    base = rec * REC2 + 4 + (r >> 1) * 16 + (c4 & 1) * 8 + (r & 1)
    if isb > 0:
        tk.gmem.store(tk.ops.mad_wide_u32(base + 2, 4, dst),
                      tk.ops.bitcast(tk.ops.pack((a0, a1), dtype="bf16"), "u32"), dtype="u32")
        tk.gmem.store(tk.ops.mad_wide_u32(base + 6, 4, dst),
                      tk.ops.bitcast(tk.ops.pack((a2, a3), dtype="bf16"), "u32"), dtype="u32")
    else:
        tk.gmem.store(tk.ops.mad_wide_u32(base, 4, dst),
                      tk.ops.bitcast(tk.ops.pack((a0 * 0.5, a1 * 0.5), dtype="f16"), "u32"), dtype="u32")
        tk.gmem.store(tk.ops.mad_wide_u32(base + 4, 4, dst),
                      tk.ops.bitcast(tk.ops.pack((a2 * 0.5, a3 * 0.5), dtype="f16"), "u32"), dtype="u32")
        zw = lane - lane
        tk.gmem.store(tk.ops.mad_wide_u32(base + XROW, 4, dst), zw, dtype="u32")
        tk.gmem.store(tk.ops.mad_wide_u32(base + XROW + 4, 4, dst), zw, dtype="u32")
        tk.gmem.store(tk.ops.mad_wide_u32(base + XROW + 2, 4, dst),
                      tk.ops.bitcast(tk.ops.pack((a0 * -16.0, a1 * -16.0), dtype="bf16"), "u32"), dtype="u32")
        tk.gmem.store(tk.ops.mad_wide_u32(base + XROW + 6, 4, dst),
                      tk.ops.bitcast(tk.ops.pack((a2 * -16.0, a3 * -16.0), dtype="bf16"), "u32"), dtype="u32")
    if lane == 0:
        tk.gmem.store(tk.ops.mad_wide_u32(rec * REC2, 4, dst), tot, dtype="f32")


@tk.kernel(maxntid=(PREP_THREADS, 1, 1))
def _v3_prep_k(ids, cnt, rows, act, nact, gu, osum, x, xp, m, nexp):
    """CTA 0 routes: per-expert pick lists and the list of work passes (expert | pass << 16, one per 4 picks of an
    active expert).  The other CTAs zero the f32 accumulators and write x as permuted records."""
    tid = tk.arch.thread_idx_x()
    bid = tk.arch.block_idx_x()
    if PDL:
        tk.pdl.arrive()
    if bid > 0:
        zero = tk.ops.zero_f32()
        z4 = (zero, zero, zero, zero)
        n1 = m * (TOPK * N1 // 4)
        stride = (ZCTAS) * PREP_THREADS
        for i in tk.range((n1 + stride - 1) // stride):
            idx = (bid - 1) * PREP_THREADS + tid + i * stride
            if idx < n1:
                tk.gmem.store(tk.ops.mad_wide_u32(idx, 16, gu), z4, dtype="f32", width=4)
        n2 = m * (N2 // 4)
        for i in tk.range((n2 + stride - 1) // stride):
            idx = (bid - 1) * PREP_THREADS + tid + i * stride
            if idx < n2:
                tk.gmem.store(tk.ops.mad_wide_u32(idx, 16, osum), z4, dtype="f32", width=4)
        nrec = m * G1
        wstride = ZCTAS * (PREP_THREADS // 32)
        wg = (bid - 1) * (PREP_THREADS // 32) + (tid >> 5)
        for i in tk.range((nrec + wstride - 1) // wstride):
            rec = wg + i * wstride
            if rec < nrec:
                t = rec // G1
                g = rec - t * G1
                xv = tk.gmem.load("u32", tk.ops.mad_wide_u32(t * K1 + g * 128 + (tid & 31) * 4, 2, x), width=2)
                _write_rec(xp, rec, tid & 31, xv[0], xv[1])
        return
    alloc = tk.shared_allocator()
    cs = alloc.allocate(tk.sv("u32", PREP_THREADS))
    ws = alloc.allocate(tk.sv("u32", 16))
    lane = tid & 31
    warp = tid >> 5
    tk.shared.store(cs.element_addr(tid), tid - tid, dtype="u32")
    tk.sync.sync_threads()
    n = m * TOPK
    for b in tk.range((n + PREP_THREADS - 1) // PREP_THREADS):
        i = b * PREP_THREADS + tid
        if i < n:
            e = tk.gmem.load("u32", tk.ops.mad_wide_u32(i, 4, ids))
            if e < nexp:
                pos = tk.shared.atomic(cs.element_addr(e), 1, dtype="u32", op="add")
                tk.gmem.store(tk.ops.mad_wide_u32(e * CAP + pos, 4, rows), i, dtype="u32")
    tk.sync.sync_threads()
    c = tk.shared.load("u32", cs.element_addr(tid))
    valid = tid < nexp
    if valid:
        tk.gmem.store(tk.ops.mad_wide_u32(tid, 4, cnt), c, dtype="u32")
    npass = (c + (TPP - 1)) // TPP
    incl = tk.warp.inclusive_scan_u32(npass, lane=lane)
    if lane == 31:
        tk.shared.store(ws.element_addr(warp), incl, dtype="u32")
    tk.sync.sync_threads()
    pre = tid - tid
    tot = tid - tid
    for w in range(PREP_THREADS // 32):
        v = tk.shared.load("u32", ws.element_addr(w))
        pre = pre + tk.ops.select(w < warp, v, 0)
        tot = tot + v
    pos = pre + incl - npass
    for j in range(CAP // TPP):
        if j < npass:
            tk.gmem.store(tk.ops.mad_wide_u32(pos + j, 4, act), tid + j * 65536, dtype="u32")
    if tid == 0:
        tk.gmem.store(nact, tot, dtype="u32")
        tk.gmem.store(tk.ops.mad_wide_u32(1, 4, nact), tid - tid, dtype="u32")
        tk.gmem.store(tk.ops.mad_wide_u32(2, 4, nact), tid - tid, dtype="u32")


@tk.inline
def _decode16(w0, w1, w2, kc):
    """3 packed words of one 16-K row -> per 16-row slice q: (T, T', R, R') (as #54's _decode16)."""
    mag, h32, h992, m6, m2 = kc
    s1 = tk.ops.shift_left(w1, 1, dtype="u32")
    s2 = tk.ops.shift_left(w2, 1, dtype="u32")
    v0 = tk.ops.shift_left(w0, 1, dtype="u32")
    v1 = tk.ops.bit_or(tk.ops.bit_and(s1, m6), tk.ops.bit_and(w0, tk.ops.bit_not(m6)))
    v2 = tk.ops.bit_or(tk.ops.bit_and(s2, m2), tk.ops.bit_and(w1, tk.ops.bit_not(m2)))
    v3 = w2
    out = []
    for v in (v0, v1, v2, v3):
        vs = tk.ops.shift_right(v, 8, dtype="u32")
        t0 = tk.ops.bit_or(tk.ops.bit_and(v, 0x00EE00EE), mag)
        t1 = tk.ops.bit_or(tk.ops.bit_and(vs, 0x00EE00EE), mag)
        r0 = tk.ops.fma(tk.ops.bitcast(t0, "f16x2"), h32, h992)
        r1 = tk.ops.fma(tk.ops.bitcast(t1, "f16x2"), h32, h992)
        out.append((t0, t1, tk.ops.bitcast(r0, "u32"), tk.ops.bitcast(r1, "u32")))
    return out


@tk.inline
def _load_scales(s, si):
    sv = tk.gmem.load("u32", tk.ops.mad_wide_u32(si, 2, s), width=4)
    sc = []
    for i in range(4):
        lo, hi = tk.ops.unpack(tk.ops.bitcast(sv[i], "bf16x2"))
        sc.append(lo)
        sc.append(hi)
    return sc


@tk.inline
def _consume(wsm, xsm, st, woff, lane, kc, h, acc):
    """One stage (RPS rows of 16 K of one 128-K group, sub-stage h) for this warp's 64-col chunk, as row pairs x 4
    slices x 2 MMAs (T in f16 with columns [x_a/2 | 0], R in bf16 with [x_b | -16 x_a]); the correction seeds
    column 2t at the group's first sub-stage."""
    wbase = wsm + (st * STAGE_W + woff + lane * 3) * 4
    xbase = xsm + (st * STAGE_X + (lane >> 2) * XROW + (lane & 3) * 4 + 4) * 4
    if h == 0:
        cbase = xsm + (st * STAGE_X + (lane & 3) * 2 * XROW) * 4
        ca = tk.shared.load("f32", cbase)
        zero = tk.ops.zero_f32()
        acc = [(ca, zero, ca, zero) for _ in range(4)]
    else:
        acc = list(acc)
    for rl in range(RPS // 2):
        rp = h * (RPS // 2) + rl
        bv = tk.shared.load("u32", xbase + rp * 64, width=4)
        dec = []
        for rr in range(2):
            wa = wbase + (2 * rl + rr) * SEG * 4
            a0 = tk.shared.load("u32", wa)
            a1 = tk.shared.load("u32", wa + 4)
            a2 = tk.shared.load("u32", wa + 8)
            dec.append(_decode16(a0, a1, a2, kc))
        for q in range(4):
            tf = (dec[0][q][0], dec[0][q][1], dec[1][q][0], dec[1][q][1])
            rf = (dec[0][q][2], dec[0][q][3], dec[1][q][2], dec[1][q][3])
            acc[q] = tk.warp.mma_m16n8k16(tf, (bv[0], bv[1]), acc[q], dtype="f16")
            acc[q] = tk.warp.mma_m16n8k16(rf, (bv[2], bv[3]), acc[q], dtype="bf16")
    return acc


def _make_gemm(is_g1, NCTAS):
    """Persistent stream-K grouped GEMV: the unit sequence is (pass, N tile, 128-K group), group fastest; CTA b takes
    units [b*U/NCTAS, (b+1)*U/NCTAS) and flushes each (pass, tile) partial with f32 reductions."""
    G = G1 if is_g1 else G2
    NT = T1 if is_g1 else T2
    WPR = WPR1 if is_g1 else WPR2
    WE = W1E if is_g1 else W2E
    NS = N1 if is_g1 else N2
    SPLIT = SPLIT1 if is_g1 else SPLIT2
    CTRI = 1 if is_g1 else 2       # work counter index in nact

    @tk.kernel(reqntid=(THREADS, 1, 1), name="v3_gemm_gu" if is_g1 else "v3_gemm_dn", max_registers=MAXREG)
    def gemm(wmap, smap, w, s, x, cnt, rows, act, nact, tw, dst):
        alloc = tk.dynamic_shared_allocator(SMEM)
        wbuf = alloc.allocate(tk.sv("u32", STAGES * STAGE_W), alignment=128)
        xbuf = alloc.allocate(tk.sv("u32", STAGES * STAGE_X), alignment=128)
        sbuf = alloc.allocate(tk.sv("u32", STAGES * SSTAGE), alignment=128)
        kbuf = alloc.allocate(tk.sv("u32", 8), alignment=16)
        mbuf = alloc.allocate(tk.sv("u32", STAGES * 4), alignment=16)
        full = alloc.mbarrier_array(STAGES)
        empty = alloc.mbarrier_array(STAGES)
        tid = tk.arch.thread_idx_x()
        bid = tk.arch.block_idx_x()
        warp = tid >> 5
        lane = tid & 31
        if PDL:
            if is_g1:
                tk.pdl.wait()     # routing, x records and zeroed accumulators of _v3_prep_k
            tk.pdl.arrive()       # gu: after its wait, so a launched swiglu implies prep complete
        npas = tk.gmem.load("u32", nact)
        if tid == 0:
            for st in range(STAGES):
                full[st].init(1)
                empty[st].init(NCW * 32)
            tk.mbarrier.init_fence()
            tk.shared.store(kbuf.element_addr(0), tk.ops.constant_i32(0x64006400), dtype="u32")
            tk.shared.store(kbuf.element_addr(1), tk.ops.constant_i32(0x04000400), dtype="u32")
            tk.shared.store(kbuf.element_addr(2), tk.ops.constant_i32(0x42E042E0), dtype="u32")
            tk.shared.store(kbuf.element_addr(4), tk.ops.constant_i32(0x66666666), dtype="u32")
            tk.shared.store(kbuf.element_addr(5), tk.ops.constant_i32(0x22222222), dtype="u32")
        tk.sync.sync_threads()
        # Dynamic chunks from a global counter (zeroed by prep): whole items first, then the last NCTAS*KS items
        # split into SPLIT chunks each, so CTAs that share an SM and run at different speeds end together.
        nitem = npas * NT
        nsm = NCTAS * KS
        nb = tk.ops.select(nitem > nsm, nitem - nsm, 0)
        nch = nb + (nitem - nb) * SPLIT
        wsm = wbuf.smem_u32
        xsm = xbuf.smem_u32
        ssm = sbuf.smem_u32
        msm = mbuf.smem_u32
        if warp == NCW:
            if lane == 0:
                seq = bid - bid
                active = seq + 1
                while active != 0:
                    c = tk.gmem.atomic(tk.ops.mad_wide_u32(CTRI, 4, nact), 1, dtype="u32", op="add", order="relaxed",
                                       scope="gpu")
                    valid = c < nch
                    big = c < nb
                    j = c - nb
                    jq = j // SPLIT
                    part = j - jq * SPLIT
                    item = tk.ops.select(big, c, nb + jq)
                    g0 = tk.ops.select(big, 0, (part * G) // SPLIT)
                    g1 = tk.ops.select(big, G, ((part + 1) * G) // SPLIT)
                    ng = tk.ops.select(valid, g1 - g0, 0)
                    item = tk.ops.select(valid, item, 0)
                    v = item // NT
                    t = item - v * NT
                    pv = tk.ops.select(valid, tk.gmem.load("u32", tk.ops.mad_wide_u32(v, 4, act)), 0)
                    e = pv & 0xFFFF
                    p = pv >> 16
                    ce = tk.ops.select(valid, tk.gmem.load("u32", tk.ops.mad_wide_u32(e, 4, cnt)), 0)
                    ntok = tk.ops.select(valid, tk.ops.minimum(ce - p * TPP, TPP, dtype="u32"), 0)
                    toks = []
                    for j in range(TPP):
                        jj = tk.ops.select(j < ntok, p * TPP + j, p * TPP)
                        pk = tk.gmem.load("u32", tk.ops.mad_wide_u32(e * CAP + jj, 4, rows))
                        toks.append(pk // TOPK if is_g1 else pk)
                    for gg in tk.range(ng):
                        g = g0 + gg
                        for h in range(SPG):
                            st = seq % STAGES
                            ph = (seq // STAGES) & 1
                            if seq >= STAGES:
                                empty.at(st, phase=ph ^ 1).wait()
                            tk.shared.store(msm + st * 16, (pv, t, ce, ng), dtype="u32", width=4)
                            sbytes = SSTAGE * 4 if h == SPG - 1 else 0
                            full[st].expect_tx(STAGE_W * 4 + sbytes + ntok * ((XROW + REC) * 4))
                            if h == SPG - 1:
                                srow = e * (G * NS) + g * NS
                                dsm = ssm + st * SSTAGE * 4
                                if is_g1 and SMAP:
                                    tk.tma.load_async(dsm, smap, (0, t, 0, e * G + g), full[st])
                                elif is_g1:
                                    tk.tma.load_async_bytes(dsm, tk.ops.mad_wide_u32(srow + t * 256, 2, s), 512, full[st])
                                    tk.tma.load_async_bytes(dsm + 512, tk.ops.mad_wide_u32(srow + I2 + t * 256, 2, s),
                                                            512, full[st])
                                else:
                                    tk.tma.load_async_bytes(dsm, tk.ops.mad_wide_u32(srow + t * 512, 2, s), 1024, full[st])
                            if TMAP:
                                d = wsm + st * (STAGE_W * 4)
                                if is_g1:
                                    tk.tma.load_async(d, wmap, (0, 2 * t, 0, e * (K1 // 16) + g * 8 + h * RPS), full[st],
                                                      cache=_WCACHE)
                                else:
                                    tk.tma.load_async(d, wmap, (0, 4 * t, e * (K2 // 16) + g * 8 + h * RPS), full[st],
                                                      cache=_WCACHE)
                            else:
                                for r in range(RPS):
                                    wrow = e * WE + (g * 8 + h * RPS + r) * WPR
                                    d = wsm + (st * STAGE_W + r * SEG) * 4
                                    if is_g1:
                                        tk.tma.load_async_bytes(d, tk.ops.mad_wide_u32(wrow + t * 384, 4, w), 1536, full[st])
                                        tk.tma.load_async_bytes(d + 1536, tk.ops.mad_wide_u32(wrow + (I2 // 64) * 96 + t * 384, 4, w),
                                                                1536, full[st])
                                    else:
                                        tk.tma.load_async_bytes(d, tk.ops.mad_wide_u32(wrow + t * 768, 4, w), 3072, full[st])
                            if PDL and not is_g1:
                                if seq == 0:
                                    tk.pdl.wait()  # the inter records of _v3_swiglu_k; weights already in flight
                            for j in range(TPP):
                                if j < ntok:
                                    tk.tma.load_async_bytes(xsm + (st * STAGE_X + 2 * j * XROW) * 4,
                                                            tk.ops.mad_wide_u32((toks[j] * G + g) * REC2, 4, x),
                                                            (XROW + REC) * 4, full[st])
                            seq = seq + 1
                    if ng == 0:
                        # no chunk left: a stage with ng = 0 tells the consumers to stop
                        st = seq % STAGES
                        ph = (seq // STAGES) & 1
                        if seq >= STAGES:
                            empty.at(st, phase=ph ^ 1).wait()
                        tk.shared.store(msm + st * 16, (pv, t, ce, ng), dtype="u32", width=4)
                        full[st].arrive()
                    active = tk.ops.select(valid, 1, 0)
        else:
            grow = lane >> 2
            if is_g1:
                isup = warp >> 2
                part = warp & 3
                woff = isup * 384 + part * 96
                sloc = isup * 256 + part * 64 + grow * 8
            else:
                woff = warp * 96
                sloc = warp * 64 + grow * 8
            kc = (tk.shared.load("u32", kbuf.element_addr(0)),
                  tk.ops.bitcast(tk.shared.load("u32", kbuf.element_addr(1)), "f16x2"),
                  tk.ops.bitcast(tk.shared.load("u32", kbuf.element_addr(2)), "f16x2"),
                  tk.shared.load("u32", kbuf.element_addr(4)), tk.shared.load("u32", kbuf.element_addr(5)))
            zero = tk.ops.zero_f32()
            seq = bid - bid
            active = seq + 1
            while active != 0:
                st0 = seq % STAGES
                full.at(st0, phase=(seq // STAGES) & 1).wait()
                mv = tk.shared.load("u32", msm + st0 * 16, width=4)
                pv = mv[0]
                t = mv[1]
                ce = mv[2]
                ng = mv[3]
                active = tk.ops.select(ng > 0, 1, 0)
                e = pv & 0xFFFF
                p = pv >> 16
                if is_g1:
                    ncol = isup * I2 + (t * 4 + part) * 64
                else:
                    ncol = (t * 8 + warp) * 64
                acc = [[zero, zero, zero, zero] for _ in range(4)]
                for gg in tk.range(ng):
                    pa = None
                    for h in range(SPG):
                        st = seq % STAGES
                        ph = (seq // STAGES) & 1
                        full.at(st, phase=ph).wait()
                        if h == SPG - 1:
                            sv = tk.shared.load("u32", ssm + (st * SSTAGE * 2 + sloc) * 2, width=4)
                        pa = _consume(wsm, xsm, st, woff, lane, kc, h, pa)
                        empty[st].arrive()
                        seq = seq + 1
                    sc = []
                    for i in range(4):
                        slo, shi = tk.ops.unpack(tk.ops.bitcast(sv[i], "bf16x2"))
                        sc.append(slo)
                        sc.append(shi)
                    for q in range(4):
                        for j in range(4):
                            acc[q][j] = tk.ops.fma(pa[q][j], sc[2 * q + (j >> 1)], acc[q][j])
                bo = p * TPP + (lane & 3)
                if bo < ce:
                    ro = tk.gmem.load("u32", tk.ops.mad_wide_u32(e * CAP + bo, 4, rows))
                    if is_g1:
                        for q in range(4):
                            for h in range(2):
                                n = ncol + 16 * q + 8 * h + grow
                                tk.gmem.reduce(tk.ops.mad_wide_u32(ro * N1 + n, 4, dst), acc[q][h * 2] + acc[q][h * 2 + 1],
                                               dtype="f32", order="relaxed", scope="gpu")
                    else:
                        wt = tk.gmem.load("f32", tk.ops.mad_wide_u32(ro, 4, tw))
                        tok = ro // TOPK
                        for q in range(4):
                            for h in range(2):
                                n = ncol + 16 * q + 8 * h + grow
                                tk.gmem.reduce(tk.ops.mad_wide_u32(tok * N2 + n, 4, dst), (acc[q][h * 2] + acc[q][h * 2 + 1]) * wt,
                                               dtype="f32", order="relaxed", scope="gpu")
    return gemm


EW_CTAS = 512


@tk.kernel(maxntid=(256, 1, 1))
def _v3_swiglu_k(gu, inter, m, limit):
    """inter records (pick, group) = silu(min(g, L)) * clamp(u, -L, L) with g, u rounded to bf16 first, result
    rounded to bf16, written in the permuted record layout with its correction."""
    if PDL:
        tk.pdl.wait()
        tk.pdl.arrive()
    total = m * (TOPK * G2)
    wstride = EW_CTAS * 8
    lane = tk.arch.thread_idx_x() & 31
    for itr in tk.range((total + wstride - 1) // wstride):
        rec = tk.arch.block_idx_x() * 8 + (tk.arch.thread_idx_x() >> 5) + itr * wstride
        if rec < total:
            r = rec // G2
            gq = rec - r * G2
            c = gq * 32 + lane
            gv = tk.gmem.load("f32", tk.ops.mad_wide_u32(r * (N1 // 4) + c, 16, gu), width=4)
            uv = tk.gmem.load("f32", tk.ops.mad_wide_u32(r * (N1 // 4) + I2 // 4 + c, 16, gu), width=4)
            res = []
            for k in range(4):
                g = tk.ops.convert(tk.ops.convert(gv[k], dtype="bf16"), dtype="f32")
                u = tk.ops.convert(tk.ops.convert(uv[k], dtype="bf16"), dtype="f32")
                g = tk.ops.minimum(g, limit)
                u = tk.ops.maximum(tk.ops.minimum(u, limit), 0.0 - limit)
                den = 1.0 + tk.ops.exp2(g * -1.4426950408889634)
                res.append(tk.ops.div(g, den) * u)
            p0 = tk.ops.bitcast(tk.ops.pack((res[0], res[1]), dtype="bf16"), "u32")
            p1 = tk.ops.bitcast(tk.ops.pack((res[2], res[3]), dtype="bf16"), "u32")
            _write_rec(inter, rec, lane, p0, p1)


@tk.kernel(maxntid=(256, 1, 1))
def _v3_cvt_k(osum, out, m):
    if PDL:
        tk.pdl.wait()
    total = m * (N2 // 4)
    stride = EW_CTAS * 256
    for itr in tk.range((total + stride - 1) // stride):
        i = tk.arch.block_idx_x() * 256 + tk.arch.thread_idx_x() + itr * stride
        if i < total:
            v = tk.gmem.load("f32", tk.ops.mad_wide_u32(i, 16, osum), width=4)
            tk.gmem.store(tk.ops.mad_wide_u32(i, 8, out),
                          (tk.ops.pack((v[0], v[1]), dtype="bf16"), tk.ops.pack((v[2], v[3]), dtype="bf16")),
                          dtype="bf16x2", width=2)


_SMS = None


def _num_sms():
    global _SMS
    if _SMS is None:
        _SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return _SMS


_PK = dict(programmatic_stream_serialization=True, pdl_producer_release="explicit") if PDL else {}


def _make_decode_prog(NCTAS):
    g1 = _make_gemm(True, NCTAS)
    g2 = _make_gemm(False, NCTAS)

    @tk.jit
    def _decode_prog(m1map, m2map, smap, x, ids, tw, w1, s1, w2, s2, cnt, rows, act, nact, gu, inter, osum, out, xp, m,
                     nexp, limit, stream):
        _v3_prep_k(ids, cnt, rows, act, nact, gu, osum, x, xp, m, nexp).launch(grid=1 + ZCTAS, block=PREP_THREADS,
                                                                               stream=stream)
        g1(m1map, smap, w1, s1, xp, cnt, rows, act, nact, tw, gu).launch(grid=NCTAS, block=THREADS, shared_memory=SMEM,
                                                                   stream=stream, **_PK)
        _v3_swiglu_k(gu, inter, m, limit).launch(grid=EW_CTAS, block=256, stream=stream, **_PK)
        g2(m2map, smap, w2, s2, inter, cnt, rows, act, nact, tw, osum).launch(grid=NCTAS, block=THREADS,
                                                                        shared_memory=SMEM, stream=stream, **_PK)
        _v3_cvt_k(osum, out, m).launch(grid=EW_CTAS, block=256, stream=stream, **_PK)
    return _decode_prog


_PROG = None
_WB = {}


def _wmaps(w1, w2, s1):
    """TMA maps over the packed weights (s32 words): gu box = 8 rows x (gate | up) x 384 words of one 256-col tile,
    dn box = 8 rows x 768 words of one 512-col tile; each lands in shared memory exactly as the row-by-row copies."""
    nexp = w1.size(0)
    key = nexp
    b = _WB.get(key)
    if b is None:
        b = (tk.TiledTensorMapBinding(dtype="s32", global_shape=(192, (I2 // 64) * 96 // 192, 2, nexp * (K1 // 16)),
                                      global_strides=(768, (I2 // 64) * 96 * 4, WPR1 * 4),
                                      box_shape=(192, 2, 2, RPS), l2_promotion_bytes=L2P),
             tk.TiledTensorMapBinding(dtype="s32", global_shape=(192, WPR2 // 192, nexp * (K2 // 16)),
                                      global_strides=(768, WPR2 * 4), box_shape=(192, 4, RPS), l2_promotion_bytes=L2P))
        b = b + (tk.TiledTensorMapBinding(dtype="bf16", global_shape=(256, T1, 2, nexp * G1),
                                          global_strides=(512, I2 * 2, N1 * 2), box_shape=(256, 1, 2, 1)),)
        _WB[key] = b
    return b[0].bind(w1.data_ptr()), b[1].bind(w2.data_ptr()), b[2].bind(s1.data_ptr())


def _get_prog(m1map, m2map, smap):
    global _PROG
    if _PROG is None:
        ptr = tk.compiler.KernelArgument.pointer(4096)
        u = tk.compiler.KernelArgument("u32", 1)
        f = tk.compiler.KernelArgument("f32", 10.0)
        _PROG = tk.compile(_make_decode_prog(CPS * _num_sms()), m1map, m2map, smap, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr,
                           ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, u, u, f, 0, target="sm103a")
    return _PROG


def moe_decode_v3(x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, w1: torch.Tensor,
                  s1: torch.Tensor, w2: torch.Tensor, s2: torch.Tensor, cnt: torch.Tensor, rows: torch.Tensor,
                  act: torch.Tensor, nact: torch.Tensor, gu: torch.Tensor, inter: torch.Tensor, osum: torch.Tensor,
                  output: torch.Tensor, xp: torch.Tensor, m: int, nexp: int, limit: float) -> None:
    m1map, m2map, smap = _wmaps(w1, w2, s1)
    _get_prog(m1map, m2map, smap)(m1map, m2map, smap, x, topk_ids, topk_weights, w1, s1, w2, s2, cnt, rows, act, nact, gu,
                            inter, osum, output, xp, m, nexp, limit, torch.cuda.current_stream().cuda_stream)


#: Each TKDSL kernel this file defines, with the arguments it writes.
KERNELS = {"moe_decode_v3": ("cnt", "rows", "act", "nact", "gu", "inter", "osum", "output", "xp")}


def _clamp_limit(self):
    cfg = getattr(self, "activation_config", None)
    limit = None if cfg is None else cfg.clamp_limit
    return 3.0e38 if limit is None else float(limit)


def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts, expert_map,
          a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input) -> None:
    m = topk_ids.size(0)
    if (m <= DECODE_MAX and expert_tokens_meta is None and expert_map is None and a1q_scale is None
            and w1.size(0) <= 512 and topk_ids.size(1) == TOPK):
        x = hidden_states.view(-1, hidden_states.size(-1))
        nexp = w1.size(0)
        dev = x.device
        cnt = torch.empty(nexp, dtype=torch.int32, device=dev)
        rows = torch.empty(nexp * CAP, dtype=torch.int32, device=dev)
        act = torch.empty(m * TOPK, dtype=torch.int32, device=dev)
        nact = torch.empty(4, dtype=torch.int32, device=dev)
        gu = torch.empty(m * TOPK, N1, dtype=torch.float32, device=dev)
        inter = torch.empty(m * TOPK * G2 * REC2, dtype=torch.int32, device=dev)
        xp = torch.empty(m * G1 * REC2, dtype=torch.int32, device=dev)
        osum = torch.empty(m, N2, dtype=torch.float32, device=dev)
        moe_decode_v3(x, topk_ids, topk_weights, w1, self.quant_config.w1_scale, w2, self.quant_config.w2_scale,
                      cnt, rows, act, nact, gu, inter, osum, output, xp, m, nexp, _clamp_limit(self))
        return
    _v2.apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
              expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input)
