"""A new vllm.model_executor.layers.fused_moe.experts.fused_humming_moe:HummingIndexedExperts.apply, put in its place by the vLLM plugin: new Python that calls new TKDSL kernels.

Decode path (m <= 64): device routing -> INT3 gate/up grouped GEMV with fused clamped SwiGLU -> INT3 down grouped GEMV
(top-k weight applied) -> top-k sum.  Weights are streamed through shared memory with bulk async copies, decoded in
registers (codes used in place as bf16 magic numbers, k8-15 activations prescaled by 1/16) and multiplied with
mma.sync m16n8k16.  Steps of 65 to PREFILL_MIN - 1 tokens run Humming; larger ones the prefill kernels.
"""

import torch
from tkdsl import nvidia as tk  # noqa: F401

from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import *  # noqa: F401,F403
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import moe_fused_mul_sum  # noqa: F401

K1, N1 = 5120, 4608
I2 = 2304
K2, N2 = 2304, 5120
TOPK = 6
CAP = 64          # max picks per expert on the decode path (m <= 64, ids distinct per token)
DECODE_MAX = 64
# The prefill kernels lose to Humming below this many tokens on a B300 (0.62x for 65-2048, 0.98x at 8192, 0.99x at
# 8283) and win above it (1.10x at 9216, 1.22x at 16384), so steps of 65 to 8447 tokens run Humming. Spec-decode
# verification steps (4 tokens per request with 3 drafts) and 4096-token prefill chunks fall in that range.
PREFILL_MIN = 8448

WPR1 = N1 * 3 // 2               # u32 words per 16-K row of w13
W1E = (K1 // 16) * WPR1          # words per expert of w13
WPR2 = N2 * 3 // 2
W2E = (K2 // 16) * WPR2

STAGES = 2
NCW = 8                          # consumer warps
THREADS = (NCW + 1) * 32
SEG = 768                        # u32 words per 16-K row of a stage (8 chunks x 96)
RPS = 8                          # 16-K weight rows per pipeline stage (8: one 128-K group, 4: half a group)
REC = 68                         # u32 words per (token, 128-K group) record: corr + 3 pad, then 64 permuted x words
XROW = 80                        # padded words per token row in the x stage (conflict-free LDS.128)
REC2 = 2 * XROW                  # u32 words per (token, group) record pair in global: type 0 at 0, type 1 at XROW
TPP = 4                          # tokens per MMA pass (two B columns per token)
RBF16 = True                     # R as bf16 128 + c_hi (bits of f16 3.5 + c_hi/512): R-MMA in bf16, bias 8x smaller
SUB_R = False                    # R -= 1024 (HADD2) so the tensor cores never accumulate the 16384*x_a bias
SUB_T = False                    # T -= 1024 as well
_TB = 0.0 if SUB_T else 1024.0
_RB = 0.0 if SUB_R else (128.0 if RBF16 else 1024.0)
CORR_A = -(_TB / 2 - 16 * _RB + 4)
CORR_B = -(_RB + 4)
STAGE_X = 8 * XROW
G1 = K1 // 128
G2 = K2 // 128
T1 = I2 // 256                   # gate/up tiles per expert (4 gate + 4 up chunks)
T2 = N2 // 512                   # down tiles per expert (8 chunks)


def _stage_w():
    return RPS * SEG


SCALE_SMEM = False               # stage each group's 512 bf16 scales through shared memory with the weights
SSTAGE = 256                     # u32 words of scales per stage


def _smem():
    return STAGES * (_stage_w() + STAGE_X + SSTAGE) * 4 + 256

PREP_THREADS = 512
CPS = 3                          # persistent CTAs per SM
MAXREG = None                    # register cap for the GEMV kernels (None: compiler's choice)
ZCTAS = 64


@tk.kernel(maxntid=(PREP_THREADS, 1, 1))
def _prep_k(ids, cnt, rows, act, nact, gu, osum, x, xp, m, nexp):
    """CTA 0 routes (per-expert pick lists, compacted active-expert list); the others zero the f32 accumulators and
    write x as permuted records (k8-15 prescaled by 1/16) with the per-(token, group) zero-point correction."""
    tid = tk.arch.thread_idx_x()
    bid = tk.arch.block_idx_x()
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
    flag = tk.ops.and_predicate(valid, c > 0)
    bal = tk.warp.ballot(flag)
    if lane == 0:
        tk.shared.store(ws.element_addr(warp), tk.ops.popcount(bal), dtype="u32")
    tk.sync.sync_threads()
    pre = tid - tid
    tot = tid - tid
    for w in range(PREP_THREADS // 32):
        v = tk.shared.load("u32", ws.element_addr(w))
        pre = pre + tk.ops.select(w < warp, v, 0)
        tot = tot + v
    lmask = tk.ops.shift_left(1, lane, dtype="u32") - 1
    pos = pre + tk.ops.popcount(tk.ops.bit_and(bal, lmask))
    if flag:
        tk.gmem.store(tk.ops.mad_wide_u32(pos, 4, act), tid, dtype="u32")
    if tid == 0:
        tk.gmem.store(nact, tot, dtype="u32")


@tk.inline
def _write_rec(dst, rec, lane, p0, p1):
    """Lane holds k = 4*lane..4*lane+3 of one 128-K group as two bf16x2 words.  Writes the f16 record pair of the
    paired-code decode: data word(rp, kq, w) = rp*16 + kq*4 + w after 4 header words, rp = 16-K row pair, kq = k-pair
    quad.  Type 0 (B column 2t): w0/w1 = x_a/2 of rows 2rp/2rp+1 (k 2kq..2kq+1), w2/w3 = x_b (k 8+2kq..).  Type 1 (B
    column 2t+1, at +XROW): w0/w1 = 0, w2/w3 = -16*x_a.  Lane 0 writes the f32 correction
    15868*sum(x_a) - 1028*sum(x_b) at word 0 of type 0 (type 1 carries none)."""
    r = lane >> 2
    c4 = lane & 3
    isb = c4 >> 1
    a0, a1 = tk.ops.unpack(tk.ops.bitcast(p0, "bf16x2"))
    a2, a3 = tk.ops.unpack(tk.ops.bitcast(p1, "bf16x2"))
    s = (a0 + a1) + (a2 + a3)
    coef = tk.ops.select(isb > 0, CORR_B, CORR_A)
    tot = tk.warp.sum(s * coef)
    base = rec * REC2 + 4 + (r >> 1) * 16 + (c4 & 1) * 8 + (r & 1)
    rdt = "bf16" if RBF16 else "f16"
    if isb > 0:
        tk.gmem.store(tk.ops.mad_wide_u32(base + 2, 4, dst),
                      tk.ops.bitcast(tk.ops.pack((a0, a1), dtype=rdt), "u32"), dtype="u32")
        tk.gmem.store(tk.ops.mad_wide_u32(base + 6, 4, dst),
                      tk.ops.bitcast(tk.ops.pack((a2, a3), dtype=rdt), "u32"), dtype="u32")
    else:
        tk.gmem.store(tk.ops.mad_wide_u32(base, 4, dst),
                      tk.ops.bitcast(tk.ops.pack((a0 * 0.5, a1 * 0.5), dtype="f16"), "u32"), dtype="u32")
        tk.gmem.store(tk.ops.mad_wide_u32(base + 4, 4, dst),
                      tk.ops.bitcast(tk.ops.pack((a2 * 0.5, a3 * 0.5), dtype="f16"), "u32"), dtype="u32")
        zw = lane - lane
        tk.gmem.store(tk.ops.mad_wide_u32(base + XROW, 4, dst), zw, dtype="u32")
        tk.gmem.store(tk.ops.mad_wide_u32(base + XROW + 4, 4, dst), zw, dtype="u32")
        tk.gmem.store(tk.ops.mad_wide_u32(base + XROW + 2, 4, dst),
                      tk.ops.bitcast(tk.ops.pack((a0 * -16.0, a1 * -16.0), dtype=rdt), "u32"), dtype="u32")
        tk.gmem.store(tk.ops.mad_wide_u32(base + XROW + 6, 4, dst),
                      tk.ops.bitcast(tk.ops.pack((a2 * -16.0, a3 * -16.0), dtype=rdt), "u32"), dtype="u32")
    if lane == 0:
        tk.gmem.store(tk.ops.mad_wide_u32(rec * REC2, 4, dst), tot, dtype="f32")


@tk.inline
def _shr(v, n):
    if not SHR_MUL:
        return tk.ops.shift_right(v, n, dtype="u32")
    lo, hi = tk.ops.unpack(tk.ops.mul_wide_u32(v, 1 << (32 - n)), dtype="f32x2")
    return tk.ops.bitcast(hi, "u32")


SHR_MUL = False


@tk.inline
def _decode16(w0, w1, w2, kc):
    """3 packed words of one 16-K row -> per 16-row slice q: (T, T', R, R') where T = f16 1024 + 2*c_lo + 32*c_hi
    holds two codes per half (T from the even bytes: rows r, T' from the odd bytes: rows r+8) and R = 1024 + c_hi
    (one f16 FMA rounding T/32 + 992).  Codes are moved to bits 1-3 of each nibble with left shifts, so the integer
    pipe only does 2 merges, 4 shifts and 8 extracts per row.  kc = (0x64006400, f16x2 1/32, f16x2 992) in opaque
    registers."""
    mag, h32, h992, hm1024, m6, m2 = kc
    s1 = tk.ops.shift_left(w1, 1, dtype="u32")
    s2 = tk.ops.shift_left(w2, 1, dtype="u32")
    v0 = tk.ops.shift_left(w0, 1, dtype="u32")
    v1 = tk.ops.bit_or(tk.ops.bit_and(s1, m6), tk.ops.bit_and(w0, tk.ops.bit_not(m6)))
    v2 = tk.ops.bit_or(tk.ops.bit_and(s2, m2), tk.ops.bit_and(w1, tk.ops.bit_not(m2)))
    v3 = w2
    out = []
    for v in (v0, v1, v2, v3):
        vs = _shr(v, 8)
        t0 = tk.ops.bit_or(tk.ops.bit_and(v, 0x00EE00EE), mag)
        t1 = tk.ops.bit_or(tk.ops.bit_and(vs, 0x00EE00EE), mag)
        r0 = tk.ops.fma(tk.ops.bitcast(t0, "f16x2"), h32, h992)
        r1 = tk.ops.fma(tk.ops.bitcast(t1, "f16x2"), h32, h992)
        if SUB_R:
            r0 = tk.ops.add(r0, hm1024)
            r1 = tk.ops.add(r1, hm1024)
        if SUB_T:
            t0 = tk.ops.bitcast(tk.ops.add(tk.ops.bitcast(t0, "f16x2"), hm1024), "u32")
            t1 = tk.ops.bitcast(tk.ops.add(tk.ops.bitcast(t1, "f16x2"), hm1024), "u32")
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
    """One stage (one 128-K group): 8 rows of 16 K for this warp's 64-col chunk, as 4 row pairs x 4 slices x 2 f16
    MMAs (T with columns [x_a/2 | 0], R with [x_b | -16 x_a]).  The zero-point correction seeds column 2t."""
    stage_w = _stage_w()
    wbase = wsm + (st * stage_w + woff + lane * 3) * 4
    xbase = xsm + (st * STAGE_X + (lane >> 2) * XROW + (lane & 3) * 4 + 4) * 4
    cbase = xsm + (st * STAGE_X + (lane & 3) * 2 * XROW) * 4
    ca = tk.shared.load("f32", cbase)
    zero = tk.ops.zero_f32()
    acc = [(ca, zero, ca, zero) for _ in range(4)]
    for rp in range(4):
        bv = tk.shared.load("u32", xbase + rp * 64, width=4)
        dec = []
        for rr in range(2):
            wa = wbase + (2 * rp + rr) * SEG * 4
            a0 = tk.shared.load("u32", wa)
            a1 = tk.shared.load("u32", wa + 4)
            a2 = tk.shared.load("u32", wa + 8)
            dec.append(_decode16(a0, a1, a2, kc))
        for q in range(4):
            tf = (dec[0][q][0], dec[0][q][1], dec[1][q][0], dec[1][q][1])
            rf = (dec[0][q][2], dec[0][q][3], dec[1][q][2], dec[1][q][3])
            acc[q] = tk.warp.mma_m16n8k16(tf, (bv[0], bv[1]), acc[q], dtype="f16")
            acc[q] = tk.warp.mma_m16n8k16(rf, (bv[2], bv[3]), acc[q], dtype="bf16" if RBF16 else "f16")
    return acc


def _make_gemm(is_g1, GPI, NCTAS):
    """Persistent split-K grouped GEMV over items (active expert, N tile, K slice of GPI 128-K groups)."""
    G = G1 if is_g1 else G2
    KS = G // GPI
    NT = T1 if is_g1 else T2
    WPR = WPR1 if is_g1 else WPR2
    WE = W1E if is_g1 else W2E
    KX = K1 if is_g1 else K2
    NS = N1 if is_g1 else N2
    PER_A = NT * KS
    STAGE_W = _stage_w()
    SMEM = _smem()
    SPG = 8 // RPS

    @tk.kernel(reqntid=(THREADS, 1, 1), name="gemm_gu" if is_g1 else "gemm_dn", max_registers=MAXREG)
    def gemm(w, s, x, cnt, rows, act, nact, tw, dst, nitems):
        alloc = tk.dynamic_shared_allocator(SMEM)
        wbuf = alloc.allocate(tk.sv("u32", STAGES * STAGE_W), alignment=128)
        xbuf = alloc.allocate(tk.sv("u32", STAGES * STAGE_X), alignment=128)
        sbuf = alloc.allocate(tk.sv("u32", STAGES * SSTAGE), alignment=128)
        kbuf = alloc.allocate(tk.sv("u32", 8), alignment=16)
        full = alloc.mbarrier_array(STAGES)
        empty = alloc.mbarrier_array(STAGES)
        tid = tk.arch.thread_idx_x()
        bid = tk.arch.block_idx_x()
        nctas = NCTAS
        warp = tid >> 5
        lane = tid & 31
        na = tk.gmem.load("u32", nact)
        if tid == 0:
            for st in range(STAGES):
                full[st].init(1)
                empty[st].init(NCW * 32)
            tk.mbarrier.init_fence()
            tk.shared.store(kbuf.element_addr(0), tk.ops.constant_i32(0x64006400), dtype="u32")
            tk.shared.store(kbuf.element_addr(1), tk.ops.constant_i32(0x04000400 if RBF16 else 0x28002800), dtype="u32")
            tk.shared.store(kbuf.element_addr(2), tk.ops.constant_i32(0x42E042E0 if RBF16 else 0x63C063C0), dtype="u32")
            tk.shared.store(kbuf.element_addr(3), tk.ops.constant_i32(0xE400E400), dtype="u32")
            tk.shared.store(kbuf.element_addr(4), tk.ops.constant_i32(0x66666666), dtype="u32")
            tk.shared.store(kbuf.element_addr(5), tk.ops.constant_i32(0x22222222), dtype="u32")
        tk.sync.sync_threads()
        nmine = (nitems - bid + nctas - 1) // nctas
        wsm = wbuf.smem_u32
        xsm = xbuf.smem_u32
        ssm = sbuf.smem_u32
        if warp == NCW:
            if lane == 0:
                seq = bid - bid
                for it in tk.range(nmine):
                    item = bid + it * nctas
                    a = item // PER_A
                    rem = item - a * PER_A
                    ks = rem // NT
                    t = rem - ks * NT
                    if a < na:
                        e = tk.gmem.load("u32", tk.ops.mad_wide_u32(a, 4, act))
                        ce = tk.gmem.load("u32", tk.ops.mad_wide_u32(e, 4, cnt))
                        for p in tk.range((ce + TPP - 1) // TPP):
                            ntok = tk.ops.minimum(ce - p * TPP, TPP, dtype="u32")
                            toks = []
                            for j in range(TPP):
                                jj = tk.ops.select(j < ntok, p * TPP + j, p * TPP)
                                pk = tk.gmem.load("u32", tk.ops.mad_wide_u32(e * CAP + jj, 4, rows))
                                toks.append(pk // TOPK if is_g1 else pk)
                            for gg in tk.range(GPI):
                                g = ks * GPI + gg
                                for h in range(SPG):
                                    if SPG == 1:
                                        xw0, xwn = 0, REC
                                    else:
                                        xw0, xwn = (0, 36) if h == 0 else (36, 32)
                                    st = seq % STAGES
                                    ph = (seq // STAGES) & 1
                                    if seq >= STAGES:
                                        empty.at(st, phase=ph ^ 1).wait()
                                    sbytes = SSTAGE * 4 if (SCALE_SMEM and h == 0) else 0
                                    full[st].expect_tx(STAGE_W * 4 + ntok * ((XROW + REC) * 4) + sbytes)
                                    if SCALE_SMEM and h == 0:
                                        srow = e * (G * NS) + g * NS
                                        dsm = ssm + st * SSTAGE * 4
                                        if is_g1:
                                            tk.tma.load_async_bytes(dsm, tk.ops.mad_wide_u32(srow + t * 256, 2, s), 512, full[st])
                                            tk.tma.load_async_bytes(dsm + 512, tk.ops.mad_wide_u32(srow + I2 + t * 256, 2, s),
                                                                    512, full[st])
                                        else:
                                            tk.tma.load_async_bytes(dsm, tk.ops.mad_wide_u32(srow + t * 512, 2, s), 1024, full[st])
                                    for r in range(RPS):
                                        wrow = e * WE + (g * 8 + h * RPS + r) * WPR
                                        d = wsm + (st * STAGE_W + r * SEG) * 4
                                        if is_g1:
                                            tk.tma.load_async_bytes(d, tk.ops.mad_wide_u32(wrow + t * 384, 4, w), 1536, full[st])
                                            tk.tma.load_async_bytes(d + 1536, tk.ops.mad_wide_u32(wrow + (I2 // 64) * 96 + t * 384, 4, w),
                                                                    1536, full[st])
                                        else:
                                            tk.tma.load_async_bytes(d, tk.ops.mad_wide_u32(wrow + t * 768, 4, w), 3072, full[st])
                                    for j in range(TPP):
                                        if j < ntok:
                                            tk.tma.load_async_bytes(xsm + (st * STAGE_X + 2 * j * XROW) * 4,
                                                                    tk.ops.mad_wide_u32((toks[j] * G + g) * REC2, 4, x),
                                                                    (XROW + REC) * 4, full[st])
                                    seq = seq + 1
        else:
            grow = lane >> 2
            kq = (lane & 3) * 2
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
                  tk.ops.bitcast(tk.shared.load("u32", kbuf.element_addr(3)), "f16x2"),
                  tk.shared.load("u32", kbuf.element_addr(4)), tk.shared.load("u32", kbuf.element_addr(5)))
            zero = tk.ops.zero_f32()
            seq = bid - bid
            for it in tk.range(nmine):
                item = bid + it * nctas
                a = item // PER_A
                rem = item - a * PER_A
                ks = rem // NT
                t = rem - ks * NT
                if a < na:
                    e = tk.gmem.load("u32", tk.ops.mad_wide_u32(a, 4, act))
                    ce = tk.gmem.load("u32", tk.ops.mad_wide_u32(e, 4, cnt))
                    if is_g1:
                        ncol = isup * I2 + (t * 4 + part) * 64
                    else:
                        ncol = (t * 8 + warp) * 64
                    sbase = e * (G * NS) + ncol + grow * 8
                    for p in tk.range((ce + TPP - 1) // TPP):
                        acc = [[zero, zero, zero, zero] for _ in range(4)]
                        for gg in tk.range(GPI):
                            g = ks * GPI + gg
                            if not SCALE_SMEM:
                                sc = _load_scales(s, sbase + g * NS)
                            pa = None
                            for h in range(SPG):
                                st = seq % STAGES
                                ph = (seq // STAGES) & 1
                                full.at(st, phase=ph).wait()
                                if SCALE_SMEM and h == 0:
                                    sv = tk.shared.load("u32", ssm + (st * SSTAGE * 2 + sloc) * 2, width=4)
                                    sc = []
                                    for i in range(4):
                                        lo, hi = tk.ops.unpack(tk.ops.bitcast(sv[i], "bf16x2"))
                                        sc.append(lo)
                                        sc.append(hi)
                                pa = _consume(wsm, xsm, st, woff, lane, kc, h, pa)
                                empty[st].arrive()
                                seq = seq + 1
                            for q in range(4):
                                for j in range(4):
                                    acc[q][j] = tk.ops.fma(pa[q][j], sc[2 * q + (j >> 1)], acc[q][j])
                        for jj in range(1):
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
def _swiglu_k(gu, inter, m, limit):
    """inter records (pick, group) = silu(min(g, L)) * clamp(u, -L, L) with g, u rounded to bf16 first (as Humming's
    bf16 gate/up), result rounded to bf16, written in the permuted record layout with its correction."""
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
def _cvt_k(osum, out, m):
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


def _make_decode_prog(GPI1, GPI2, NCTAS):
    g1 = _make_gemm(True, GPI1, NCTAS)
    g2 = _make_gemm(False, GPI2, NCTAS)

    @tk.jit
    def _decode_prog(x, ids, tw, w1, s1, w2, s2, cnt, rows, act, nact, gu, inter, osum, out, xp, m, nexp, limit,
                     it1, it2, stream):
        _prep_k(ids, cnt, rows, act, nact, gu, osum, x, xp, m, nexp).launch(grid=1 + ZCTAS, block=PREP_THREADS,
                                                                            stream=stream)
        g1(w1, s1, xp, cnt, rows, act, nact, tw, gu, it1).launch(grid=NCTAS, block=THREADS, shared_memory=_smem(),
                                                                 stream=stream)
        _swiglu_k(gu, inter, m, limit).launch(grid=EW_CTAS, block=256, stream=stream)
        g2(w2, s2, inter, cnt, rows, act, nact, tw, osum, it2).launch(grid=NCTAS, block=THREADS, shared_memory=_smem(),
                                                                      stream=stream)
        _cvt_k(osum, out, m).launch(grid=EW_CTAS, block=256, stream=stream)
    return _decode_prog


_PROGS = {}


def _get_prog(GPI1, GPI2):
    key = (GPI1, GPI2)
    p = _PROGS.get(key)
    if p is None:
        ptr = tk.compiler.KernelArgument.pointer(4096)
        u = tk.compiler.KernelArgument("u32", 1)
        f = tk.compiler.KernelArgument("f32", 10.0)
        p = tk.compile(_make_decode_prog(GPI1, GPI2, CPS * _num_sms()), ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr,
                       ptr, ptr, ptr, ptr, ptr, ptr, u, u, f, u, u, 0, target="sm103a")
        _PROGS[key] = p
    return p


def _plan(m, nexp):
    """Split-K granularity by token count (host value): groups of 128 K per work item."""
    if m <= 2:
        gpi1, gpi2 = 2, 2
    elif m <= 8:
        gpi1, gpi2 = 5, 3
    else:
        gpi1, gpi2 = 20, 6
    amax = min(m * TOPK, nexp)
    it1 = amax * T1 * (G1 // gpi1)
    it2 = amax * T2 * (G2 // gpi2)
    return gpi1, gpi2, it1, it2


def moe_decode(x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, w1: torch.Tensor,
               s1: torch.Tensor, w2: torch.Tensor, s2: torch.Tensor, cnt: torch.Tensor, rows: torch.Tensor,
               act: torch.Tensor, nact: torch.Tensor, gu: torch.Tensor, inter: torch.Tensor, osum: torch.Tensor,
               output: torch.Tensor, xp: torch.Tensor, m: int, nexp: int, limit: float) -> None:
    gpi1, gpi2, it1, it2 = _plan(m, nexp)
    _get_prog(gpi1, gpi2)(x, topk_ids, topk_weights, w1, s1, w2, s2, cnt, rows, act, nact, gu, inter, osum, output,
                          xp, m, nexp, limit, it1, it2, torch.cuda.current_stream().cuda_stream)


#: Each TKDSL kernel this file defines, with the arguments it writes: {"kernel": ("out", ...)}.
_DECODE_KERNELS = {"moe_decode": ("cnt", "rows", "act", "nact", "gu", "inter", "osum", "output", "xp")}


def _clamp_limit(self):
    cfg = getattr(self, "activation_config", None)
    limit = None if cfg is None else cfg.clamp_limit
    return 3.0e38 if limit is None else float(limit)



from vllm.model_executor.layers.fused_moe.activation import MoEActivation as _MoEActivation

E = 384
TOPK = 6
HID = 5120
INTER = 2304
N13 = 2 * INTER

MB = 128          # token rows per tile (tcgen05 M)
NT = 256          # weight rows per tile (tcgen05 N)
KB = 64           # K per stage (one 128-byte swizzle row)
PSTAGES = 4
EPI_WARPS = 4
ISSUER_WARP = 4
TMA_WARP = 5
DQ0 = 6
DQ_WARPS = 8
UPW = 16 // DQ_WARPS   # 16x64 units per dequant warp per stage
RSTEP = DQ_WARPS // 4
PTHREADS = (DQ0 + DQ_WARPS) * 32
NSM = 148         # persistent grid: one CTA per SM of the B300
INVALID = 0xFFFFFFFF
LOG2E = 1.4426950408889634

A_TILE = tk.st("bf16", MB, KB, swizzle_bytes=128)
B_TILE = tk.st("bf16", NT, KB, swizzle_bytes=128)
ACC_TILE = tk.tt("f32", MB, NT)
FRAG = tk.rt("f32", 32, 32)
TMEM_SLOT = tk.tiles.sv("u32", 16)
RAW_W = 6144      # packed INT3 words of one stage (4 rows of 16 K x 4 chunks of 64 N x 384 B)
RAW_S = 512       # one scale row of the 256 N columns (bf16)
RAW_BYTES = RAW_W + RAW_S
SHARED_BYTES = PSTAGES * (A_TILE.byte_size + B_TILE.byte_size + RAW_BYTES) + 2048
MMA = tk.mma_descriptor(d=ACC_TILE, a=A_TILE, b=B_TILE, m=MB, n=NT, ab_dtype=tk.BF16, d_dtype=tk.F32,
                        cta_group=1, trans_b=False)


def _max_rows(m):
    return ((m * TOPK + 255) // 256 + E) * 256


# ----------------------------------------------------------------------------------------------------------------
# Routing: one CTA counts picks per expert, lays out 128-row tiles per expert, and scatters pick ids into rows.
# ----------------------------------------------------------------------------------------------------------------
@tk.kernel(reqntid=(1024, 1, 1))
def _proute_k(ids, meta, picks, tile_e, tile_c, npick):
    alloc = tk.shared_allocator()
    cnt = alloc.allocate(tk.tiles.sv("u32", 512))
    tst = alloc.allocate(tk.tiles.sv("u32", 512))
    off = alloc.allocate(tk.tiles.sv("u32", 512))
    t = tk.arch.thread_idx_x()
    if t < 512:
        tk.shared.store(cnt.element_addr(t), t - t, dtype="u32")
    tk.sync.sync_threads()
    i = t
    while i < npick:
        e = tk.gmem.load("u32", tk.ops.mad_wide_u32(i, 4, ids))
        tk.shared.atomic(cnt.element_addr(e), 1, dtype="u32")
        i = i + 1024
    tk.sync.sync_threads()
    if t == 0:
        acc = t - t
        racc = t - t
        for ex in tk.range(E):
            c = tk.shared.load("u32", cnt.element_addr(ex))
            tk.shared.store(tst.element_addr(ex), acc, dtype="u32")
            tk.shared.store(off.element_addr(ex), racc, dtype="u32")
            acc = acc + ((c + 255) >> 8)
            racc = racc + c
        tk.gmem.store(meta, acc, dtype="u32")
    tk.sync.sync_threads()
    if t < E:
        c = tk.shared.load("u32", cnt.element_addr(t))
        ts = tk.shared.load("u32", tst.element_addr(t))
        ro = tk.shared.load("u32", off.element_addr(t))
        nt = (c + 255) >> 8
        for j in tk.range(nt):
            tk.gmem.store(tk.ops.mad_wide_u32(ts + j, 4, tile_e), t, dtype="u32")
            rem = c - j * 256
            tk.gmem.store(tk.ops.mad_wide_u32(ts + j, 4, tile_c),
                          tk.ops.bit_or(tk.ops.shift_left(ro + j * 256, 9, dtype="u32"),
                                        tk.ops.minimum(rem, 256, dtype="u32")), dtype="u32")
    tk.sync.sync_threads()
    if t < E:
        tk.shared.store(cnt.element_addr(t), t - t, dtype="u32")
    tk.sync.sync_threads()
    i = t
    while i < npick:
        e = tk.gmem.load("u32", tk.ops.mad_wide_u32(i, 4, ids))
        p = tk.shared.atomic(cnt.element_addr(e), 1, dtype="u32")
        ro = tk.shared.load("u32", off.element_addr(e))
        tk.gmem.store(tk.ops.mad_wide_u32(ro + p, 4, picks), i, dtype="u32")
        i = i + 1024


# Gather token rows into expert-sorted order (compact: row r holds the token of pick picks[r], r < npick).
PGATHER_CTAS = 148 * 8


@tk.kernel(reqntid=(128, 1, 1))
def _pgather_k(x, picks, xs, meta):
    t = tk.arch.thread_idx_x()
    row = tk.arch.block_idx_x()
    nvalid = tk.gmem.load("u32", tk.ops.mad_wide_u32(1, 4, meta))
    while row < nvalid:
        p = tk.gmem.load("u32", tk.ops.mad_wide_u32(row, 4, picks))
        tok = p // TOPK
        for j in range(5):
            v = tk.gmem.load("u32", tk.ops.mad_wide_u32(tok * (HID // 8) + j * 128 + t, 16, x), width=4)
            tk.gmem.store(tk.ops.mad_wide_u32(row * (HID // 8) + j * 128 + t, 16, xs), v, width=4)
        row = row + PGATHER_CTAS


# Top-k sum of the weighted down rows: out[t] = sum_j down[t*6+j].
PREDUCE_CTAS = 148 * 8


@tk.kernel(reqntid=(256, 1, 1))
def _preduce_k(down, ids, out, m):
    i = tk.arch.block_idx_x() * 256 + tk.arch.thread_idx_x()
    while i < m * (HID // 8):
        t = i // (HID // 8)
        p = i - t * (HID // 8)
        acc = [tk.ops.zero_f32() for _ in range(8)]
        for j in range(TOPK):
            ok = tk.gmem.load("u32", tk.ops.mad_wide_u32(t * TOPK + j, 4, ids)) < E
            v = tk.gmem.load("u32", tk.ops.mad_wide_u32((t * TOPK + j) * (HID // 8) + p, 16, down), width=4)
            for q in range(4):
                vq = tk.ops.select(ok, v[q], 0)
                lo, hi = tk.ops.unpack(tk.ops.bitcast(vq, "bf16x2"))
                acc[2 * q] = acc[2 * q] + lo
                acc[2 * q + 1] = acc[2 * q + 1] + hi
        words = tuple(tk.ops.bitcast(tk.ops.pack((acc[2 * q], acc[2 * q + 1]), dtype="bf16"), "u32") for q in range(4))
        tk.gmem.store(tk.ops.mad_wide_u32(i, 16, out), words, width=4)
        i = i + PREDUCE_CTAS * 256


# ----------------------------------------------------------------------------------------------------------------
# Grouped GEMM with in-smem INT3 dequant.
# ----------------------------------------------------------------------------------------------------------------
@tk.inline
def _bcast_lo(w):
    return tk.ops.bit_or(tk.ops.bit_and(w, 0xFFFF), tk.ops.shift_left(w, 16))


@tk.inline
def _bcast_hi(w):
    return tk.ops.bit_or(tk.ops.shift_right(w, 16, dtype="u32"), tk.ops.bit_and(w, 0xFFFF0000))


@tk.inline
def _decode_unit(w0, w1, w2, sc, sc16, neg, neg16, m7, m70, mag):
    """3 words of one lane -> per type t (m16 block) the 4 bf16x2 registers (g0, g2, g1, g3) in stmatrix order.
    Nibble-parallel: codes of groups 1/3 stay at bits 4-6 (128+16c, offset 192, scale s/16)."""
    vv = (
        w0,
        tk.ops.bit_or(tk.ops.bit_and(w1, 0x33333333), tk.ops.bit_and(tk.ops.shift_right(w0, 1, dtype="u32"), 0xCCCCCCCC)),
        tk.ops.bit_or(tk.ops.bit_and(w2, 0x11111111), tk.ops.bit_and(tk.ops.shift_right(w1, 1, dtype="u32"), 0xEEEEEEEE)),
        tk.ops.shift_right(w2, 1, dtype="u32"),
    )
    regs = []
    for t in range(4):
        v = vv[t]
        vs = tk.ops.shift_right(v, 8, dtype="u32")
        g0 = tk.ops.bit_or(tk.ops.bit_and(v, m7), mag)
        g1 = tk.ops.bit_or(tk.ops.bit_and(v, m70), mag)
        g2 = tk.ops.bit_or(tk.ops.bit_and(vs, m7), mag)
        g3 = tk.ops.bit_or(tk.ops.bit_and(vs, m70), mag)
        r0 = tk.ops.mul(tk.ops.add(tk.ops.bitcast(g0, "bf16x2"), neg), sc[t][0])
        r1 = tk.ops.mul(tk.ops.add(tk.ops.bitcast(g1, "bf16x2"), neg16), sc16[t][0])
        r2 = tk.ops.mul(tk.ops.add(tk.ops.bitcast(g2, "bf16x2"), neg), sc[t][1])
        r3 = tk.ops.mul(tk.ops.add(tk.ops.bitcast(g3, "bf16x2"), neg16), sc16[t][1])
        regs.append((tk.ops.bitcast(r0, "u32"), tk.ops.bitcast(r2, "u32"), tk.ops.bitcast(r1, "u32"),
                     tk.ops.bitcast(r3, "u32")))
    return regs


def _pmake_gemm(kind):
    gate_up = kind == 1
    K = HID if gate_up else INTER
    N = N13 if gate_up else HID
    NCH = N // 64
    WPR = NCH * 96              # u32 words per 16-K row of one expert
    EW = (K // 16) * WPR        # u32 words per expert
    ES = (K // 128) * N         # scales per expert
    KBLKS = K // KB
    NBLK = INTER // 128 if gate_up else HID // NT
    OUT_N = INTER if gate_up else HID

    @tk.kernel(reqntid=(PTHREADS, 1, 1), min_blocks_per_sm=1, name=f"p_gemm{kind}")
    def kernel(a_map, w, s, meta, tile_e, tile_c, picks, tw, out, limit, alpha, beta):
        allocator = tk.dynamic_shared_allocator(SHARED_BYTES, alignment=1024)
        inputs = allocator.stages(PSTAGES, A=A_TILE, B=B_TILE)
        rawbuf = allocator.allocate(tk.sv("u32", PSTAGES * RAW_BYTES // 4), alignment=128)
        tmem_slot = allocator.allocate(TMEM_SLOT)
        kconst = allocator.allocate(tk.sv("u32", 4), alignment=16)
        tmem_ready = allocator.mbarrier()
        a_full = allocator.mbarrier_array(PSTAGES)
        b_full = allocator.mbarrier_array(PSTAGES)
        stage_done = allocator.mbarrier_array(PSTAGES)
        acc_ready = allocator.mbarrier_array(2)
        acc_empty = allocator.mbarrier_array(2)
        thread = tk.arch.thread_idx_x()
        warp = tk.uniform.warp_id()
        lane = tk.arch.lane_id()
        cta = tk.uniform.wrap(tk.arch.block_idx_x())
        tasks = tk.uniform.wrap(tk.gmem.load("u32", meta) * NBLK)
        if warp == TMA_WARP:
            if tk.arch.elect_leader():
                tmem_ready.init(1)
                tk.shared.store(kconst.element_addr(0), tk.ops.constant_i32(0x00070007), dtype="u32")
                tk.shared.store(kconst.element_addr(1), tk.ops.constant_i32(0x00700070), dtype="u32")
            a_full.init(1)
            b_full.init(DQ_WARPS * 32)
            stage_done.init(1)
            acc_ready.init(1)
            acc_empty.init(EPI_WARPS)
            tk.mbarrier.init_fence()
        tk.sync.sync_threads()
        if thread < 32:
            arena = tk.tensor_allocator(cta_group=1, cols=512)
            arena.provision(tmem_slot.smem_u32)
            if thread == 0:
                tmem_ready.arrive()
        if warp < EPI_WARPS:
            tmem_ready.wait()
            base = tk.shared.load("u32", tmem_slot.smem_u32)
            regions = tk.tensor_allocator(base, cta_group=1, cols=512).stages(2, ACC_TILE)
            local_row = warp * 32 + lane
            task = cta
            count = tk.uniform.wrap(0)
            nalpha = alpha * (-LOG2E)
            while task < tasks:
                tile, nb = tk.uniform.divmod_const(task, NBLK)
                cnt = tk.gmem.load("u32", tk.ops.mad_wide_u32(tile, 4, tile_c))
                row = tile * MB + local_row
                valid = local_row < cnt
                region = count & 1
                acc_ready.at(region, phase=count >> 1 & 1).wait()
                accumulator = regions[region]
                if gate_up:
                    for cc in range(4):
                        fg = tk.tmem.load(FRAG, accumulator, col_offset=cc * 32, superlane=warp << 5)
                        fu = tk.tmem.load(FRAG, accumulator, col_offset=128 + cc * 32, superlane=warp << 5)
                        tk.tmem.wait_load()
                        gv = tuple(fg.regs)
                        uv = tuple(fu.regs)
                        res = []
                        for j in range(32):
                            g = tk.ops.convert(tk.ops.convert(gv[j], dtype="bf16"), dtype="f32")
                            u = tk.ops.convert(tk.ops.convert(uv[j], dtype="bf16"), dtype="f32")
                            g = tk.ops.minimum(g, limit)
                            u = tk.ops.maximum(tk.ops.minimum(u, limit), -limit)
                            den = 1.0 + tk.ops.exp2(g * nalpha)
                            res.append(tk.ops.div(g, den) * (u + beta))
                        words = tuple(tk.ops.bitcast(tk.ops.pack((res[2 * i], res[2 * i + 1]), dtype="bf16"), "u32")
                                      for i in range(16))
                        if valid:
                            address = tk.ops.mad_wide_u32(row * (OUT_N // 8) + nb * 16 + cc * 4, 16, out)
                            for c4 in range(4):
                                tk.gmem.store(address, words[4 * c4:4 * c4 + 4], width=4, byte_offset=16 * c4)
                else:
                    pick = tk.gmem.load("u32", tk.ops.mad_wide_u32(row, 4, picks))
                    pk = tk.ops.select(valid, pick, 0)
                    wt = tk.gmem.load("f32", tk.ops.mad_wide_u32(pk, 4, tw))
                    for cc in range(8):
                        fv = tk.tmem.load(FRAG, accumulator, col_offset=cc * 32, superlane=warp << 5)
                        tk.tmem.wait_load()
                        vv = tuple(fv.regs)
                        words = tuple(tk.ops.bitcast(tk.ops.pack((vv[2 * i] * wt, vv[2 * i + 1] * wt), dtype="bf16"), "u32")
                                      for i in range(16))
                        if valid:
                            address = tk.ops.mad_wide_u32(pk * (OUT_N // 8) + nb * 32 + cc * 4, 16, out)
                            for c4 in range(4):
                                tk.gmem.store(address, words[4 * c4:4 * c4 + 4], width=4, byte_offset=16 * c4)
                tk.sync.fence_before_thread_sync()
                if tk.arch.elect_leader():
                    acc_empty.at(region).arrive()
                task = tk.uniform.wrap(task + NSM)
                count = tk.uniform.wrap(count + 1)
        elif warp == ISSUER_WARP:
            tmem_ready.wait()
            base = tk.shared.load("u32", tmem_slot.smem_u32)
            regions = tk.tensor_allocator(base, cta_group=1, cols=512).stages(2, ACC_TILE)
            task = cta
            count = tk.uniform.wrap(0)
            iteration = tk.uniform.wrap(0)
            while task < tasks:
                region = count & 1
                if count >= 2:
                    acc_empty.at(region, phase=count >> 1 & 1 ^ 1).wait()
                for k in tk.uniform.range(KBLKS, kind="dynamic", unroll=1):
                    stage_round, stage = tk.uniform.divmod_const(iteration, PSTAGES)
                    a_full.at(stage, phase=stage_round & 1).wait()
                    b_full.at(stage, phase=stage_round & 1).wait()
                    if tk.arch.elect_leader():
                        tk.group(4).mma_ABt(regions[region], inputs[stage].A, inputs[stage].B, MMA,
                                            a_desc=inputs[stage].A.matrix_descriptor(mn_major=False),
                                            b_desc=inputs[stage].B.matrix_descriptor(mn_major=False),
                                            accumulate=k != 0, dst_cta_mask=None)
                        tk.mma_commit(MMA, stage_done.at(stage), dst_cta_mask=None)
                    iteration = tk.uniform.wrap(iteration + 1)
                if tk.arch.elect_leader():
                    tk.mma_commit(MMA, acc_ready.at(region), dst_cta_mask=None)
                task = tk.uniform.wrap(task + NSM)
                count = tk.uniform.wrap(count + 1)
        elif warp == TMA_WARP:
            stage = tk.uniform.ring_index(0, PSTAGES)
            phase = tk.uniform.wrap(0)
            iteration = tk.uniform.wrap(0)
            task = cta
            while task < tasks:
                tile, nb = tk.uniform.divmod_const(task, NBLK)
                a_row = tile * MB
                e = tk.gmem.load("u32", tk.ops.mad_wide_u32(tile, 4, tile_e))
                wb = e * EW
                sb = e * ES
                for k in tk.uniform.range(KBLKS, kind="dynamic", unroll=1):
                    if iteration >= PSTAGES:
                        stage_done.at(stage, phase=phase ^ 1).wait()
                    if tk.arch.elect_leader():
                        a_full.at(stage).expect_tx(A_TILE.byte_size + RAW_BYTES)
                        tk.tma.load_async(inputs[stage].A, a_map, (k * KB, a_row), a_full.at(stage),
                                          cache=tk.tma.CachePolicy.EVICT_LAST)
                        dst = rawbuf.smem_u32 + stage * RAW_BYTES
                        for rl in range(4):
                            wrow = wb + (k * 4 + rl) * WPR
                            if gate_up:
                                tk.tma.load_async_bytes(dst + rl * 1536, tk.ops.mad_wide_u32(wrow + nb * 192, 4, w),
                                                        768, a_full.at(stage))
                                tk.tma.load_async_bytes(dst + rl * 1536 + 768,
                                                        tk.ops.mad_wide_u32(wrow + (INTER // 64 + nb * 2) * 96, 4, w),
                                                        768, a_full.at(stage))
                            else:
                                tk.tma.load_async_bytes(dst + rl * 1536, tk.ops.mad_wide_u32(wrow + nb * 384, 4, w),
                                                        1536, a_full.at(stage))
                        srow = sb + (k >> 1) * N
                        if gate_up:
                            tk.tma.load_async_bytes(dst + RAW_W, tk.ops.mad_wide_u32(srow + nb * 128, 2, s),
                                                    256, a_full.at(stage))
                            tk.tma.load_async_bytes(dst + RAW_W + 256, tk.ops.mad_wide_u32(srow + INTER + nb * 128, 2, s),
                                                    256, a_full.at(stage))
                        else:
                            tk.tma.load_async_bytes(dst + RAW_W, tk.ops.mad_wide_u32(srow + nb * 256, 2, s),
                                                    512, a_full.at(stage))
                    phase = tk.uniform.wrap(tk.ops.select(stage == PSTAGES - 1, phase ^ 1, phase))
                    stage = tk.uniform.wrap_inc(stage, PSTAGES)
                    iteration = tk.uniform.wrap(iteration + 1)
                task = tk.uniform.wrap(task + NSM)
        else:
            d = warp - DQ0
            ci = d & 3
            rsub = d >> 2
            grow = lane >> 2
            mi = lane >> 3
            mj = lane & 7
            neg = tk.ops.bitcast(tk.ops.constant_i32(0xC304C304), "bf16x2")
            neg16 = tk.ops.bitcast(tk.ops.constant_i32(0xC340C340), "bf16x2")
            sixteenth = tk.ops.bitcast(tk.ops.constant_i32(0x3D803D80), "bf16x2")
            stage = tk.uniform.ring_index(0, PSTAGES)
            phase = tk.uniform.wrap(0)
            iteration = tk.uniform.wrap(0)
            task = cta
            woff = rawbuf.smem_u32 + ci * 384 + lane * 12
            soff = rawbuf.smem_u32 + RAW_W + ci * 128 + grow * 16
            m7 = tk.shared.load("u32", kconst.element_addr(0))
            m70 = tk.shared.load("u32", kconst.element_addr(1))
            mag = tk.ops.constant_i32(0x43004300)
            boffs = []
            for rr in range(UPW):
                rl = rsub + rr * RSTEP
                chunk = rl * 2 + (mi >> 1)
                boffs.append((ci * 8 + (mi & 1)) * 1024 + mj * 128 + ((chunk ^ mj) << 4))
            while task < tasks:
                for kb in tk.uniform.range(KBLKS, kind="dynamic", unroll=1):
                    a_full.at(stage, phase=phase).wait()
                    so = stage * RAW_BYTES
                    sv = tk.shared.load("u32", soff + so, width=4)
                    wds = []
                    for rr in range(UPW):
                        wa = woff + so + (rsub + rr * RSTEP) * 1536
                        wds.append((tk.shared.load("u32", wa), tk.shared.load("u32", wa + 4),
                                    tk.shared.load("u32", wa + 8)))
                    sc = [(tk.ops.bitcast(_bcast_lo(sv[q]), "bf16x2"), tk.ops.bitcast(_bcast_hi(sv[q]), "bf16x2"))
                          for q in range(4)]
                    sc16 = [(tk.ops.mul(sc[q][0], sixteenth), tk.ops.mul(sc[q][1], sixteenth)) for q in range(4)]
                    bbase = inputs[stage].B.smem_u32
                    for rr in range(UPW):
                        rl = rsub + rr * RSTEP
                        regs = _decode_unit(wds[rr][0], wds[rr][1], wds[rr][2], sc, sc16, neg, neg16, m7, m70, mag)
                        ub = bbase + boffs[rr]
                        for t in range(4):
                            addr = ub + 2 * t * 1024
                            tk.shared.store_matrix(getattr(addr, "value", addr), tuple(regs[t]), count=4)
                    tk.sync.fence_async_shared()
                    b_full.at(stage).arrive()
                    phase = tk.uniform.wrap(tk.ops.select(stage == PSTAGES - 1, phase ^ 1, phase))
                    stage = tk.uniform.wrap_inc(stage, PSTAGES)
                    iteration = tk.uniform.wrap(iteration + 1)
                task = tk.uniform.wrap(task + NSM)
        tk.sync.sync_threads()
        if thread < 32:
            base = tk.shared.load("u32", tmem_slot.smem_u32)
            tk.tensor_allocator(base, cta_group=1, cols=512).deprovision()

    return kernel


# ---- 2-CTA pair GEMM: M=256 token rows per pair (128 per CTA), N=256, each CTA dequantizes its 128-N half of B.
MT = 256                       # token rows per routing tile (one CTA pair)
PST2 = 6
NPAIR = NSM // 2
BH_TILE = tk.st("bf16", 128, KB, swizzle_bytes=128)
RAW2_W = 3072                  # 4 rows of 16 K x 2 chunks of 64 N x 384 B
RAW2 = RAW2_W + 256            # + one scale row of 128 N
SHARED2 = PST2 * (A_TILE.byte_size + BH_TILE.byte_size + RAW2) + 2048
MMA2 = tk.mma_descriptor(d=ACC_TILE, a=A_TILE, b=BH_TILE, m=256, n=NT, ab_dtype=tk.BF16, d_dtype=tk.F32,
                         cta_group=2, trans_b=False)


def _pmake_gemm_pair(kind):
    gate_up = kind == 1
    K = HID if gate_up else INTER
    N = N13 if gate_up else HID
    NCH = N // 64
    WPR = NCH * 96
    EW = (K // 16) * WPR
    ES = (K // 128) * N
    KBLKS = K // KB
    NBLK = INTER // 128 if gate_up else HID // NT
    OUT_N = INTER if gate_up else HID

    @tk.kernel(reqntid=(PTHREADS, 1, 1), min_blocks_per_sm=1, name=f"p2_gemm{kind}")
    def kernel(a_map, w, s, meta, tile_e, tile_c, picks, tw, out, limit, alpha, beta):
        allocator = tk.dynamic_shared_allocator(SHARED2, alignment=1024)
        inputs = allocator.stages(PST2, A=A_TILE, B=BH_TILE)
        rawbuf = allocator.allocate(tk.sv("u32", PST2 * RAW2 // 4), alignment=128)
        tmem_slot = allocator.allocate(TMEM_SLOT)
        kconst = allocator.allocate(tk.sv("u32", 4), alignment=16)
        tmem_ready = allocator.mbarrier()
        a_full = allocator.mbarrier_array(PST2)
        r_full = allocator.mbarrier_array(PST2)
        b_full = allocator.mbarrier_array(PST2)
        stage_done = allocator.mbarrier_array(PST2)
        acc_ready = allocator.mbarrier_array(2)
        acc_empty = allocator.mbarrier_array(2)
        drained = allocator.mbarrier(cluster_scope=True)
        thread = tk.arch.thread_idx_x()
        warp = tk.uniform.warp_id()
        lane = tk.arch.lane_id()
        rank = tk.cluster.rank()
        pair = tk.uniform.wrap(tk.arch.block_idx_x() >> 1)
        tasks = tk.uniform.wrap(tk.gmem.load("u32", meta) * NBLK)
        if warp == TMA_WARP:
            if tk.arch.elect_leader():
                tmem_ready.init(1)
                drained.init(1)
                tk.shared.store(kconst.element_addr(0), tk.ops.constant_i32(0x00070007), dtype="u32")
                tk.shared.store(kconst.element_addr(1), tk.ops.constant_i32(0x00700070), dtype="u32")
            a_full.init(1)
            r_full.init(1)
            b_full.init(2 * DQ_WARPS)
            stage_done.init(1)
            acc_ready.init(1)
            acc_empty.init(2 * EPI_WARPS)
            tk.mbarrier.init_fence()
        tk.sync.cluster_arrive()
        tk.sync.cluster_wait()
        if thread < 32:
            tk.tensor_allocator(cta_group=2, cols=512).provision(tmem_slot.smem_u32)
            if thread == 0:
                tmem_ready.arrive()
        if warp < EPI_WARPS:
            tmem_ready.wait()
            base = tk.shared.load("u32", tmem_slot.smem_u32)
            local_row = rank * 128 + warp * 32 + lane
            task = pair
            count = tk.uniform.wrap(0)
            nalpha = alpha * (-LOG2E)
            while task < tasks:
                tile, nb = tk.uniform.divmod_const(task, NBLK)
                tcv = tk.gmem.load("u32", tk.ops.mad_wide_u32(tile, 4, tile_c))
                cnt = tcv & 511
                row = (tcv >> 9) + local_row
                valid = local_row < cnt
                region = count & 1
                acc_ready.at(region, phase=count >> 1 & 1).wait()
                tk.sync.fence_after_thread_sync()
                accumulator = ACC_TILE.handle(base + region * 256)
                if gate_up:
                    for cc in range(4):
                        fg = tk.tmem.load(FRAG, accumulator, col_offset=cc * 32, superlane=warp << 5)
                        fu = tk.tmem.load(FRAG, accumulator, col_offset=128 + cc * 32, superlane=warp << 5)
                        tk.tmem.wait_load()
                        gv = tuple(fg.regs)
                        uv = tuple(fu.regs)
                        res = []
                        for j in range(32):
                            g = tk.ops.convert(tk.ops.convert(gv[j], dtype="bf16"), dtype="f32")
                            u = tk.ops.convert(tk.ops.convert(uv[j], dtype="bf16"), dtype="f32")
                            g = tk.ops.minimum(g, limit)
                            u = tk.ops.maximum(tk.ops.minimum(u, limit), -limit)
                            den = 1.0 + tk.ops.exp2(g * nalpha)
                            res.append(tk.ops.div(g, den) * (u + beta))
                        words = tuple(tk.ops.bitcast(tk.ops.pack((res[2 * i], res[2 * i + 1]), dtype="bf16"), "u32")
                                      for i in range(16))
                        if valid:
                            address = tk.ops.mad_wide_u32(row * (OUT_N // 8) + nb * 16 + cc * 4, 16, out)
                            for c4 in range(4):
                                tk.gmem.store(address, words[4 * c4:4 * c4 + 4], width=4, byte_offset=16 * c4)
                else:
                    pick = tk.gmem.load("u32", tk.ops.mad_wide_u32(row, 4, picks))
                    pk = tk.ops.select(valid, pick, 0)
                    wt = tk.gmem.load("f32", tk.ops.mad_wide_u32(pk, 4, tw))
                    for cc in range(8):
                        fv = tk.tmem.load(FRAG, accumulator, col_offset=cc * 32, superlane=warp << 5)
                        tk.tmem.wait_load()
                        vv = tuple(fv.regs)
                        words = tuple(tk.ops.bitcast(tk.ops.pack((vv[2 * i] * wt, vv[2 * i + 1] * wt), dtype="bf16"), "u32")
                                      for i in range(16))
                        if valid:
                            address = tk.ops.mad_wide_u32(pk * (OUT_N // 8) + nb * 32 + cc * 4, 16, out)
                            for c4 in range(4):
                                tk.gmem.store(address, words[4 * c4:4 * c4 + 4], width=4, byte_offset=16 * c4)
                tk.sync.fence_before_thread_sync()
                if tk.arch.elect_leader():
                    acc_empty.at(region).arrive_remote(0)
                task = tk.uniform.wrap(task + NPAIR)
                count = tk.uniform.wrap(count + 1)
        elif warp == ISSUER_WARP:
            if rank == 0:
                tmem_ready.wait()
                base = tk.shared.load("u32", tmem_slot.smem_u32)
                task = pair
                count = tk.uniform.wrap(0)
                iteration = tk.uniform.wrap(0)
                while task < tasks:
                    region = count & 1
                    if count >= 2:
                        acc_empty.at(region, phase=count >> 1 & 1 ^ 1).wait()
                        tk.sync.fence_after_thread_sync()
                    accumulator = ACC_TILE.handle(base + region * 256)
                    for k in tk.uniform.range(KBLKS, kind="dynamic", unroll=1):
                        stage_round, stage = tk.uniform.divmod_const(iteration, PST2)
                        a_full.at(stage, phase=stage_round & 1).wait()
                        b_full.at(stage, phase=stage_round & 1).wait()
                        if tk.arch.elect_leader():
                            tk.group(4).mma_ABt(accumulator, inputs[stage].A, inputs[stage].B, MMA2,
                                                a_desc=inputs[stage].A.matrix_descriptor(mn_major=False),
                                                b_desc=inputs[stage].B.matrix_descriptor(mn_major=False),
                                                accumulate=k != 0, dst_cta_mask=3)
                            tk.mma_commit(MMA2, stage_done.at(stage), dst_cta_mask=3)
                        iteration = tk.uniform.wrap(iteration + 1)
                    if tk.arch.elect_leader():
                        tk.mma_commit(MMA2, acc_ready.at(region), dst_cta_mask=3)
                    task = tk.uniform.wrap(task + NPAIR)
                    count = tk.uniform.wrap(count + 1)
        elif warp == TMA_WARP:
            stage = tk.uniform.ring_index(0, PST2)
            phase = tk.uniform.wrap(0)
            iteration = tk.uniform.wrap(0)
            task = pair
            while task < tasks:
                tile, nb = tk.uniform.divmod_const(task, NBLK)
                tcv = tk.gmem.load("u32", tk.ops.mad_wide_u32(tile, 4, tile_c))
                a_row = (tcv >> 9) + rank * 128
                e = tk.gmem.load("u32", tk.ops.mad_wide_u32(tile, 4, tile_e))
                if gate_up:
                    c0 = rank * (INTER // 64) + nb * 2
                else:
                    c0 = nb * 4 + rank * 2
                wb = e * EW + c0 * 96
                sb = e * ES + c0 * 64
                for k in tk.uniform.range(KBLKS, kind="dynamic", unroll=1):
                    if iteration >= PST2:
                        stage_done.at(stage, phase=phase ^ 1).wait()
                    if tk.arch.elect_leader():
                        if rank == 0:
                            a_full.at(stage).expect_tx(2 * A_TILE.byte_size)
                        tk.tma.load_async(inputs[stage].A, a_map, (k * KB, a_row), a_full.at(stage),
                                          cache=tk.tma.CachePolicy.EVICT_LAST, cluster_scope=True, cta_group=2,
                                          dst_mbar_cta=0)
                        r_full.at(stage).expect_tx(RAW2)
                        dst = rawbuf.smem_u32 + stage * RAW2
                        for rl in range(4):
                            tk.tma.load_async_bytes(dst + rl * 768, tk.ops.mad_wide_u32(wb + (k * 4 + rl) * WPR, 4, w),
                                                    768, r_full.at(stage))
                        tk.tma.load_async_bytes(dst + RAW2_W, tk.ops.mad_wide_u32(sb + (k >> 1) * N, 2, s),
                                                256, r_full.at(stage))
                    phase = tk.uniform.wrap(tk.ops.select(stage == PST2 - 1, phase ^ 1, phase))
                    stage = tk.uniform.wrap_inc(stage, PST2)
                    iteration = tk.uniform.wrap(iteration + 1)
                task = tk.uniform.wrap(task + NPAIR)
        elif warp >= DQ0:
            d = warp - DQ0
            ci = d & 1
            rl = d >> 1
            grow = lane >> 2
            mi = lane >> 3
            mj = lane & 7
            neg = tk.ops.bitcast(tk.ops.constant_i32(0xC304C304), "bf16x2")
            neg16 = tk.ops.bitcast(tk.ops.constant_i32(0xC340C340), "bf16x2")
            sixteenth = tk.ops.bitcast(tk.ops.constant_i32(0x3D803D80), "bf16x2")
            stage = tk.uniform.ring_index(0, PST2)
            phase = tk.uniform.wrap(0)
            task = pair
            woff = rawbuf.smem_u32 + rl * 768 + ci * 384 + lane * 12
            soff = rawbuf.smem_u32 + RAW2_W + ci * 128 + grow * 16
            m7 = tk.shared.load("u32", kconst.element_addr(0))
            m70 = tk.shared.load("u32", kconst.element_addr(1))
            mag = tk.ops.constant_i32(0x43004300)
            chunk = rl * 2 + (mi >> 1)
            boff = (ci * 8 + (mi & 1)) * 1024 + mj * 128 + ((chunk ^ mj) << 4)
            while task < tasks:
                for kb in tk.uniform.range(KBLKS, kind="dynamic", unroll=1):
                    r_full.at(stage, phase=phase).wait()
                    so = stage * RAW2
                    sv = tk.shared.load("u32", soff + so, width=4)
                    wa = woff + so
                    w0 = tk.shared.load("u32", wa)
                    w1 = tk.shared.load("u32", wa + 4)
                    w2 = tk.shared.load("u32", wa + 8)
                    sc = [(tk.ops.bitcast(_bcast_lo(sv[q]), "bf16x2"), tk.ops.bitcast(_bcast_hi(sv[q]), "bf16x2"))
                          for q in range(4)]
                    sc16 = [(tk.ops.mul(sc[q][0], sixteenth), tk.ops.mul(sc[q][1], sixteenth)) for q in range(4)]
                    regs = _decode_unit(w0, w1, w2, sc, sc16, neg, neg16, m7, m70, mag)
                    ub = inputs[stage].B.smem_u32 + boff
                    for t in range(4):
                        addr = ub + 2 * t * 1024
                        tk.shared.store_matrix(getattr(addr, "value", addr), tuple(regs[t]), count=4)
                    tk.sync.fence_async_shared()
                    tk.sync.sync_warp()
                    if tk.arch.elect_leader():
                        b_full.at(stage).arrive_remote(0, semantics="relaxed")
                    phase = tk.uniform.wrap(tk.ops.select(stage == PST2 - 1, phase ^ 1, phase))
                    stage = tk.uniform.wrap_inc(stage, PST2)
                task = tk.uniform.wrap(task + NPAIR)
        tk.sync.sync_threads()
        if thread < 32:
            if thread == 0:
                drained.arrive_remote(rank ^ 1)
            drained.wait()
            base = tk.shared.load("u32", tmem_slot.smem_u32)
            tk.tensor_allocator(base, cta_group=2, cols=512).deprovision()

    return kernel


_GEMM1 = _pmake_gemm_pair(1)
_GEMM2 = _pmake_gemm_pair(2)


# ---- Swap-AB grouped GEMM: A = 128 dequantized weight rows (M), B = up to 256 routed token rows (N, runtime, 32-step).
WM = 128
TN = 256
SST = 3                        # token/A stages in shared memory
RST = 16                       # raw INT3 word stages (own ring, own producer warp)
DEC2 = 1                       # codes at bits 4-6 read as 8+c (no scale/16 multiplies)
ABL = ""                       # ablation switches (timing experiments only); empty in production
RAW_WARP = DQ0 + DQ_WARPS
STHREADS = (RAW_WARP + 1) * 32
WA_TILE = tk.st("bf16", WM, KB, swizzle_bytes=128)
XB_TILE = tk.st("bf16", TN, KB, swizzle_bytes=128)
XBOX = tk.st("bf16", 32, KB, swizzle_bytes=128)
SRAW_W = 3072                  # 4 rows of 16 K x 2 chunks of 64 N x 384 B
SRAW = SRAW_W + 256            # + 128 scales
SSHARED = SST * (WA_TILE.byte_size + XB_TILE.byte_size) + RST * SRAW + 2048
SACC = tk.tt("f32", WM, TN)
XBJ = [tk.st("bf16", 32 * (j + 1), KB, swizzle_bytes=128) for j in range(8)]
SACCJ = [tk.tt("f32", WM, 32 * (j + 1)) for j in range(8)]
SMMAJ = [tk.mma_descriptor(d=SACCJ[j], a=WA_TILE, b=XBJ[j], m=WM, n=32 * (j + 1), ab_dtype=tk.BF16, d_dtype=tk.F32,
                           cta_group=1, trans_b=False) for j in range(8)]


@tk.kernel(reqntid=(1024, 1, 1))
def _sroute_k(ids, meta, picks, tile_e, tile_c, npick):
    """Per-expert compact row ranges (no padding): expert e owns rows off_e .. off_e + c_e - 1 of the sorted layout.
    Tiles: every full 256-row tile first (heaviest work first), then one remainder tile per expert.  tile_c packs
    (first row << 9) | row count."""
    alloc = tk.shared_allocator()
    cnt = alloc.allocate(tk.tiles.sv("u32", 512))
    fst = alloc.allocate(tk.tiles.sv("u32", 512))
    rst = alloc.allocate(tk.tiles.sv("u32", 512))
    off = alloc.allocate(tk.tiles.sv("u32", 512))
    t = tk.arch.thread_idx_x()
    if t < 512:
        tk.shared.store(cnt.element_addr(t), t - t, dtype="u32")
    tk.sync.sync_threads()
    i = t
    while i < npick:
        e = tk.gmem.load("u32", tk.ops.mad_wide_u32(i, 4, ids))
        if e < E:
            tk.shared.atomic(cnt.element_addr(e), 1, dtype="u32")
        i = i + 1024
    tk.sync.sync_threads()
    if t == 0:
        facc = t - t
        oacc = t - t
        for ex in tk.range(E):
            c = tk.shared.load("u32", cnt.element_addr(ex))
            tk.shared.store(fst.element_addr(ex), facc, dtype="u32")
            tk.shared.store(off.element_addr(ex), oacc, dtype="u32")
            facc = facc + (c >> 8)
            oacc = oacc + c
        racc = facc
        for ex in tk.range(E):
            c = tk.shared.load("u32", cnt.element_addr(ex))
            tk.shared.store(rst.element_addr(ex), racc, dtype="u32")
            racc = racc + tk.ops.select((c & 255) != 0, 1, 0)
        tk.gmem.store(meta, racc, dtype="u32")
        tk.gmem.store(tk.ops.mad_wide_u32(1, 4, meta), oacc, dtype="u32")
    tk.sync.sync_threads()
    if t < E:
        c = tk.shared.load("u32", cnt.element_addr(t))
        fs = tk.shared.load("u32", fst.element_addr(t))
        rs = tk.shared.load("u32", rst.element_addr(t))
        ro = tk.shared.load("u32", off.element_addr(t))
        nf = c >> 8
        for j in tk.range(nf):
            tk.gmem.store(tk.ops.mad_wide_u32(fs + j, 4, tile_e), t, dtype="u32")
            tk.gmem.store(tk.ops.mad_wide_u32(fs + j, 4, tile_c),
                          tk.ops.bit_or(tk.ops.shift_left(ro + j * 256, 9, dtype="u32"), 256), dtype="u32")
        rem = c & 255
        if rem != 0:
            tk.gmem.store(tk.ops.mad_wide_u32(rs, 4, tile_e), t, dtype="u32")
            tk.gmem.store(tk.ops.mad_wide_u32(rs, 4, tile_c),
                          tk.ops.bit_or(tk.ops.shift_left(ro + nf * 256, 9, dtype="u32"), rem), dtype="u32")
    tk.sync.sync_threads()
    if t < E:
        tk.shared.store(cnt.element_addr(t), t - t, dtype="u32")
    tk.sync.sync_threads()
    i = t
    while i < npick:
        e = tk.gmem.load("u32", tk.ops.mad_wide_u32(i, 4, ids))
        if e < E:
            p = tk.shared.atomic(cnt.element_addr(e), 1, dtype="u32")
            ro = tk.shared.load("u32", off.element_addr(e))
            tk.gmem.store(tk.ops.mad_wide_u32(ro + p, 4, picks), i, dtype="u32")
        i = i + 1024


@tk.inline
def _decode_unit2(w0, w1, w2, sc, neg, neg8, m7, m70, mag, mag8):
    """3 words of one lane -> per m16 block t the 4 bf16x2 registers in stmatrix order.
    Codes at bits 0-2 read as 128+c (minus 132), codes at bits 4-6 as 8+c (minus 12): both give c-4 exactly."""
    vv = (
        w0,
        tk.ops.bit_or(tk.ops.bit_and(w1, 0x33333333), tk.ops.bit_and(tk.ops.shift_right(w0, 1, dtype="u32"), 0xCCCCCCCC)),
        tk.ops.bit_or(tk.ops.bit_and(w2, 0x11111111), tk.ops.bit_and(tk.ops.shift_right(w1, 1, dtype="u32"), 0xEEEEEEEE)),
        tk.ops.shift_right(w2, 1, dtype="u32"),
    )
    regs = []
    for t in range(4):
        v = vv[t]
        vs = tk.ops.shift_right(v, 8, dtype="u32")
        g0 = tk.ops.bit_or(tk.ops.bit_and(v, m7), mag)
        g1 = tk.ops.bit_or(tk.ops.bit_and(v, m70), mag8)
        g2 = tk.ops.bit_or(tk.ops.bit_and(vs, m7), mag)
        g3 = tk.ops.bit_or(tk.ops.bit_and(vs, m70), mag8)
        r0 = tk.ops.mul(tk.ops.add(tk.ops.bitcast(g0, "bf16x2"), neg), sc[t][0])
        r1 = tk.ops.mul(tk.ops.add(tk.ops.bitcast(g1, "bf16x2"), neg8), sc[t][0])
        r2 = tk.ops.mul(tk.ops.add(tk.ops.bitcast(g2, "bf16x2"), neg), sc[t][1])
        r3 = tk.ops.mul(tk.ops.add(tk.ops.bitcast(g3, "bf16x2"), neg8), sc[t][1])
        regs.append((tk.ops.bitcast(r0, "u32"), tk.ops.bitcast(r2, "u32"), tk.ops.bitcast(r1, "u32"),
                     tk.ops.bitcast(r3, "u32")))
    return regs


@tk.inline
def _bcast2(w):
    lo = tk.ops.bitcast(tk.ops.bitcast(tk.ops.pack((w, w), dtype="f16", low_bits=True), "u32"), "bf16x2")
    return (lo, tk.ops.bitcast(_bcast_hi(w), "bf16x2"))


def _smake_gemm(kind):
    gate_up = kind == 1
    K = HID if gate_up else INTER
    N = N13 if gate_up else HID
    NCH = N // 64
    WPR = NCH * 96
    EW = (K // 16) * WPR
    ES = (K // 128) * N
    KBLKS = K // KB
    NBLK = N // WM
    OUT_N = INTER if gate_up else HID

    @tk.kernel(reqntid=(STHREADS, 1, 1), min_blocks_per_sm=1, name=f"s_gemm{kind}")
    def kernel(x_map, w, s, meta, tile_e, tile_c, picks, tw, out, limit, alpha, beta):
        allocator = tk.dynamic_shared_allocator(SSHARED, alignment=1024)
        inputs = allocator.stages(SST, A=WA_TILE, B=XB_TILE)
        rawbuf = allocator.allocate(tk.sv("u32", RST * SRAW // 4), alignment=128)
        tmem_slot = allocator.allocate(TMEM_SLOT)
        kconst = allocator.allocate(tk.sv("u32", 4), alignment=16)
        tmem_ready = allocator.mbarrier()
        x_full = allocator.mbarrier_array(SST)
        r_full = allocator.mbarrier_array(RST)
        r_empty = allocator.mbarrier_array(RST)
        w_full = allocator.mbarrier_array(SST)
        stage_done = allocator.mbarrier_array(SST)
        acc_ready = allocator.mbarrier_array(2)
        acc_empty = allocator.mbarrier_array(2)
        thread = tk.arch.thread_idx_x()
        warp = tk.uniform.warp_id()
        lane = tk.arch.lane_id()
        cta = tk.uniform.wrap(tk.arch.block_idx_x())
        tasks = tk.uniform.wrap(tk.gmem.load("u32", meta) * NBLK)
        if warp == TMA_WARP:
            if tk.arch.elect_leader():
                tmem_ready.init(1)
                tk.shared.store(kconst.element_addr(0), tk.ops.constant_i32(0x00070007), dtype="u32")
                tk.shared.store(kconst.element_addr(1), tk.ops.constant_i32(0x00700070), dtype="u32")
            x_full.init(1)
            r_full.init(1)
            r_empty.init(DQ_WARPS * 32)
            w_full.init(DQ_WARPS * 32)
            stage_done.init(1)
            acc_ready.init(1)
            acc_empty.init(EPI_WARPS)
            tk.mbarrier.init_fence()
        tk.sync.sync_threads()
        if thread < 32:
            arena = tk.tensor_allocator(cta_group=1, cols=512)
            arena.provision(tmem_slot.smem_u32)
            if thread == 0:
                tmem_ready.arrive()
        if warp < EPI_WARPS:
            tmem_ready.wait()
            base = tk.shared.load("u32", tmem_slot.smem_u32)
            task = cta
            count = tk.uniform.wrap(0)
            nalpha = alpha * (-LOG2E)
            upside = (lane & 16) != 0
            if gate_up:
                jcol = (lane & 15) + warp * 16
            else:
                ncol = lane + warp * 32
            while task < tasks:
                tile, nb = tk.uniform.divmod_const(task, NBLK)
                tcv = tk.uniform.wrap(tk.gmem.load("u32", tk.ops.mad_wide_u32(tile, 4, tile_c)))
                cnt = tk.uniform.wrap(tcv & 511)
                rst0 = tcv >> 9
                region = count & 1
                acc_ready.at(region, phase=count >> 1 & 1).wait()
                tk.sync.fence_after_thread_sync()
                accumulator = SACC.handle(base + region * 256)
                for cc in range(8):
                    if cc * 32 < cnt:
                        fv = tk.tmem.load(FRAG, accumulator, col_offset=cc * 32, superlane=warp << 5)
                        tk.tmem.wait_load()
                        vv = tuple(fv.regs)
                        if gate_up:
                            lbase = cc * 32 + tk.ops.select(upside, 1, 0)
                            obase = nb * 64 + jcol
                            for i in range(16):
                                send = tk.ops.select(upside, vv[2 * i], vv[2 * i + 1])
                                recv = tk.warp.shuffle_xor(send, 16)
                                g = tk.ops.select(upside, recv, vv[2 * i])
                                u = tk.ops.select(upside, vv[2 * i + 1], recv)
                                g = tk.ops.convert(tk.ops.convert(g, dtype="bf16"), dtype="f32")
                                u = tk.ops.convert(tk.ops.convert(u, dtype="bf16"), dtype="f32")
                                g = tk.ops.minimum(g, limit)
                                u = tk.ops.maximum(tk.ops.minimum(u, limit), -limit)
                                den = 1.0 + tk.ops.exp2(g * nalpha)
                                r = tk.ops.div(g, den) * (u + beta)
                                lt = lbase + 2 * i
                                ok = lt < cnt
                                tok = rst0 + lt
                                tk.gmem.store(tk.ops.mad_wide_u32(tok * OUT_N + obase, 2, out),
                                              tk.ops.convert(r, dtype="bf16"), dtype="bf16", mask=ok)
                        else:
                            lrow = cc * 32 + lane
                            lok = lrow < cnt
                            pick = tk.gmem.load("u32", tk.ops.mad_wide_u32(rst0 + lrow, 4, picks))
                            pk = tk.ops.select(lok, pick, 0)
                            wt = tk.gmem.load("f32", tk.ops.mad_wide_u32(pk, 4, tw))
                            ocol = nb * 128 + ncol
                            for i in range(32):
                                pi = tk.warp.shuffle_idx(pk, i)
                                wi = tk.warp.shuffle_idx(wt, i)
                                ok = cc * 32 + i < cnt
                                tk.gmem.store(tk.ops.mad_wide_u32(pi * OUT_N + ocol, 2, out),
                                              tk.ops.convert(vv[i] * wi, dtype="bf16"), dtype="bf16", mask=ok)
                tk.sync.fence_before_thread_sync()
                if tk.arch.elect_leader():
                    acc_empty.at(region).arrive()
                task = tk.uniform.wrap(task + NSM)
                count = tk.uniform.wrap(count + 1)
        elif warp == ISSUER_WARP:
            tmem_ready.wait()
            base = tk.shared.load("u32", tmem_slot.smem_u32)
            task = cta
            count = tk.uniform.wrap(0)
            iteration = tk.uniform.wrap(0)
            while task < tasks:
                tile, nb = tk.uniform.divmod_const(task, NBLK)
                cnt = tk.uniform.wrap(tk.gmem.load("u32", tk.ops.mad_wide_u32(tile, 4, tile_c)) & 511)
                nbox = tk.uniform.wrap((cnt + 31) >> 5)
                region = count & 1
                if count >= 2:
                    acc_empty.at(region, phase=count >> 1 & 1 ^ 1).wait()
                    tk.sync.fence_after_thread_sync()
                for k in tk.uniform.range(KBLKS, kind="dynamic", unroll=1):
                    stage_round, stage = tk.uniform.divmod_const(iteration, SST)
                    x_full.at(stage, phase=stage_round & 1).wait()
                    w_full.at(stage, phase=stage_round & 1).wait()
                    if tk.arch.elect_leader():
                        for j in range(8):
                            if nbox == j + 1 and "nomma" not in ABL:
                                bt = XBJ[j].handle(inputs[stage].B.smem_u32)
                                tk.group(4).mma_ABt(SACCJ[j].handle(base + region * 256), inputs[stage].A, bt,
                                                    SMMAJ[j],
                                                    a_desc=inputs[stage].A.matrix_descriptor(mn_major=False),
                                                    b_desc=bt.matrix_descriptor(mn_major=False),
                                                    accumulate=k != 0, dst_cta_mask=None)
                        if "nocommit" in ABL:
                            stage_done.at(stage).arrive()
                        else:
                            tk.mma_commit(SMMAJ[7], stage_done.at(stage), dst_cta_mask=None)
                    iteration = tk.uniform.wrap(iteration + 1)
                if tk.arch.elect_leader():
                    tk.mma_commit(SMMAJ[7], acc_ready.at(region), dst_cta_mask=None)
                task = tk.uniform.wrap(task + NSM)
                count = tk.uniform.wrap(count + 1)
        elif warp == TMA_WARP:
            stage = tk.uniform.ring_index(0, SST)
            phase = tk.uniform.wrap(0)
            iteration = tk.uniform.wrap(0)
            task = cta
            while task < tasks:
                tile, nb = tk.uniform.divmod_const(task, NBLK)
                e = tk.gmem.load("u32", tk.ops.mad_wide_u32(tile, 4, tile_e))
                tcv = tk.uniform.wrap(tk.gmem.load("u32", tk.ops.mad_wide_u32(tile, 4, tile_c)))
                cnt = tk.uniform.wrap(tcv & 511)
                nbox = tk.uniform.wrap((cnt + 31) >> 5)
                xrow = tcv >> 9
                if gate_up:
                    wb = e * EW + nb * 96
                    wu = e * EW + (INTER // 64 + nb) * 96
                    sb = e * ES + nb * 64
                    su = e * ES + INTER + nb * 64
                else:
                    wb = e * EW + nb * 192
                    sb = e * ES + nb * 128
                for k in tk.uniform.range(KBLKS, kind="dynamic", unroll=1):
                    if iteration >= SST:
                        stage_done.at(stage, phase=phase ^ 1).wait()
                    if tk.arch.elect_leader():
                        if "nox" in ABL:
                            x_full.at(stage).arrive()
                        else:
                            x_full.at(stage).expect_tx(nbox * 4096)
                        xb = inputs[stage].B.smem_u32
                        for j in range(8):
                            if j < nbox and "nox" not in ABL:
                                tk.tma.load_async(XBOX.handle(xb + j * 4096), x_map, (k * KB, xrow + j * 32),
                                                  x_full.at(stage), cache=tk.tma.CachePolicy.EVICT_LAST)
                    phase = tk.uniform.wrap(tk.ops.select(stage == SST - 1, phase ^ 1, phase))
                    stage = tk.uniform.wrap_inc(stage, SST)
                    iteration = tk.uniform.wrap(iteration + 1)
                task = tk.uniform.wrap(task + NSM)
        elif warp == RAW_WARP:
            rstage = tk.uniform.ring_index(0, RST)
            rphase = tk.uniform.wrap(0)
            iteration = tk.uniform.wrap(0)
            task = cta
            while task < tasks:
                tile, nb = tk.uniform.divmod_const(task, NBLK)
                e = tk.gmem.load("u32", tk.ops.mad_wide_u32(tile, 4, tile_e))
                if "l2w" in ABL:
                    e = e & 1
                if gate_up:
                    wb = e * EW + nb * 96
                    wu = e * EW + (INTER // 64 + nb) * 96
                    sb = e * ES + nb * 64
                    su = e * ES + INTER + nb * 64
                else:
                    wb = e * EW + nb * 192
                    sb = e * ES + nb * 128
                for k in tk.uniform.range(KBLKS, kind="dynamic", unroll=1):
                    if iteration >= RST:
                        r_empty.at(rstage, phase=rphase ^ 1).wait()
                    if tk.arch.elect_leader() and "noraw" in ABL:
                        r_full.at(rstage).arrive()
                    if tk.arch.elect_leader() and "noraw" not in ABL:
                        r_full.at(rstage).expect_tx(SRAW)
                        dst = rawbuf.smem_u32 + rstage * SRAW
                        for rl in range(4):
                            if gate_up:
                                tk.tma.load_async_bytes(dst + rl * 768, tk.ops.mad_wide_u32(wb + (k * 4 + rl) * WPR, 4, w),
                                                        384, r_full.at(rstage))
                                tk.tma.load_async_bytes(dst + rl * 768 + 384,
                                                        tk.ops.mad_wide_u32(wu + (k * 4 + rl) * WPR, 4, w),
                                                        384, r_full.at(rstage))
                            else:
                                tk.tma.load_async_bytes(dst + rl * 768, tk.ops.mad_wide_u32(wb + (k * 4 + rl) * WPR, 4, w),
                                                        768, r_full.at(rstage))
                        if gate_up:
                            tk.tma.load_async_bytes(dst + SRAW_W, tk.ops.mad_wide_u32(sb + (k >> 1) * N, 2, s),
                                                    128, r_full.at(rstage))
                            tk.tma.load_async_bytes(dst + SRAW_W + 128, tk.ops.mad_wide_u32(su + (k >> 1) * N, 2, s),
                                                    128, r_full.at(rstage))
                        else:
                            tk.tma.load_async_bytes(dst + SRAW_W, tk.ops.mad_wide_u32(sb + (k >> 1) * N, 2, s),
                                                    256, r_full.at(rstage))
                    rphase = tk.uniform.wrap(tk.ops.select(rstage == RST - 1, rphase ^ 1, rphase))
                    rstage = tk.uniform.wrap_inc(rstage, RST)
                    iteration = tk.uniform.wrap(iteration + 1)
                task = tk.uniform.wrap(task + NSM)
        elif warp >= DQ0 and warp < RAW_WARP:
            d = warp - DQ0
            ci = d & 1
            rl = d >> 1
            grow = lane >> 2
            mi = lane >> 3
            mj = lane & 7
            neg = tk.ops.bitcast(tk.ops.constant_i32(0xC304C304), "bf16x2")
            neg16 = tk.ops.bitcast(tk.ops.constant_i32(0xC340C340), "bf16x2")
            sixteenth = tk.ops.bitcast(tk.ops.constant_i32(0x3D803D80), "bf16x2")
            stage = tk.uniform.ring_index(0, SST)
            phase = tk.uniform.wrap(0)
            rstage = tk.uniform.ring_index(0, RST)
            rphase = tk.uniform.wrap(0)
            iteration = tk.uniform.wrap(0)
            task = cta
            woff = rawbuf.smem_u32 + rl * 768 + ci * 384 + lane * 12
            soff = rawbuf.smem_u32 + SRAW_W + ci * 128 + grow * 16
            m7 = tk.shared.load("u32", kconst.element_addr(0))
            m70 = tk.shared.load("u32", kconst.element_addr(1))
            mag = tk.ops.constant_i32(0x43004300)
            mag8 = tk.ops.constant_i32(0x41004100)
            neg8 = tk.ops.bitcast(tk.ops.constant_i32(0xC140C140), "bf16x2")
            chunk = rl * 2 + (mi >> 1)
            if gate_up:
                boff = (ci * 2 + (mi & 1)) * 1024 + mj * 128 + ((chunk ^ mj) << 4)
                TSTEP = 4096
            else:
                boff = (ci * 8 + (mi & 1)) * 1024 + mj * 128 + ((chunk ^ mj) << 4)
                TSTEP = 2048
            while task < tasks:
                for kb in tk.uniform.range(KBLKS, kind="dynamic", unroll=1):
                    r_full.at(rstage, phase=rphase).wait()
                    so = rstage * SRAW
                    sv = tk.shared.load("u32", soff + so, width=4)
                    wa = woff + so
                    w0 = tk.shared.load("u32", wa)
                    w1 = tk.shared.load("u32", wa + 4)
                    w2 = tk.shared.load("u32", wa + 8)
                    r_empty.at(rstage).arrive()
                    if "nodq" in ABL:
                        regs = [(w0, w1, w2, sv[0])] * 4
                    elif DEC2:
                        sc = [_bcast2(sv[q]) for q in range(4)]
                        regs = _decode_unit2(w0, w1, w2, sc, neg, neg8, m7, m70, mag, mag8)
                    else:
                        sc = [(tk.ops.bitcast(_bcast_lo(sv[q]), "bf16x2"), tk.ops.bitcast(_bcast_hi(sv[q]), "bf16x2"))
                              for q in range(4)]
                        sc16 = [(tk.ops.mul(sc[q][0], sixteenth), tk.ops.mul(sc[q][1], sixteenth)) for q in range(4)]
                        regs = _decode_unit(w0, w1, w2, sc, sc16, neg, neg16, m7, m70, mag)
                    if iteration >= SST:
                        stage_done.at(stage, phase=phase ^ 1).wait()
                    ub = inputs[stage].A.smem_u32 + boff
                    if "nostsm" not in ABL:
                        for t in range(4):
                            addr = ub + t * TSTEP
                            tk.shared.store_matrix(getattr(addr, "value", addr), tuple(regs[t]), count=4)
                    if "nofence" not in ABL:
                        tk.sync.fence_async_shared()
                    w_full.at(stage).arrive()
                    phase = tk.uniform.wrap(tk.ops.select(stage == SST - 1, phase ^ 1, phase))
                    stage = tk.uniform.wrap_inc(stage, SST)
                    rphase = tk.uniform.wrap(tk.ops.select(rstage == RST - 1, rphase ^ 1, rphase))
                    rstage = tk.uniform.wrap_inc(rstage, RST)
                    iteration = tk.uniform.wrap(iteration + 1)
                task = tk.uniform.wrap(task + NSM)
        tk.sync.sync_threads()
        if thread < 32:
            base = tk.shared.load("u32", tmem_slot.smem_u32)
            tk.tensor_allocator(base, cta_group=1, cols=512).deprovision()

    return kernel


_SGEMM1 = _smake_gemm(1)
_SGEMM2 = _smake_gemm(2)




@tk.jit
def _prefill_prog(x, ids, tw, w1, s1, w2, s2, meta, picks, tile_e, tile_c, xs, inter, out, xs_map, in_map,
                  m, npick, limit, alpha, beta, stream):
    # xs (gathered rows) and the down rows share workspace2: xs is dead once GEMM1 has finished.
    _proute_k(ids, meta, picks, tile_e, tile_c, npick).launch(grid=1, block=1024, stream=stream)
    _pgather_k(x, picks, xs, meta).launch(grid=PGATHER_CTAS, block=128, stream=stream)
    _GEMM1(xs_map, w1, s1, meta, tile_e, tile_c, picks, tw, inter, limit, alpha, beta).launch(
        grid=2 * NPAIR, block=PTHREADS, cluster=2, shared_memory=SHARED2, stream=stream)
    _GEMM2(in_map, w2, s2, meta, tile_e, tile_c, picks, tw, xs, limit, alpha, beta).launch(
        grid=2 * NPAIR, block=PTHREADS, cluster=2, shared_memory=SHARED2, stream=stream)
    _preduce_k(xs, ids, out, m).launch(grid=PREDUCE_CTAS, block=256, stream=stream)


@tk.jit
def _prefill_swap_prog(x, ids, tw, w1, s1, w2, s2, meta, picks, tile_e, tile_c, xs, inter, out, xs_map, in_map,
                       m, npick, limit, alpha, beta, stream):
    _sroute_k(ids, meta, picks, tile_e, tile_c, npick).launch(grid=1, block=1024, stream=stream)
    _pgather_k(x, picks, xs, meta).launch(grid=PGATHER_CTAS, block=128, stream=stream)
    _SGEMM1(xs_map, w1, s1, meta, tile_e, tile_c, picks, tw, inter, limit, alpha, beta).launch(
        grid=NSM, block=STHREADS, shared_memory=SSHARED, stream=stream)
    _SGEMM2(in_map, w2, s2, meta, tile_e, tile_c, picks, tw, xs, limit, alpha, beta).launch(
        grid=NSM, block=STHREADS, shared_memory=SSHARED, stream=stream)
    _preduce_k(xs, ids, out, m).launch(grid=PREDUCE_CTAS, block=256, stream=stream)


SWAP_MAX_M = 16384          # single-CTA swap-AB GEMM for every m: the 2-CTA pair path needs a release.cluster handoff that costs 2x (15.4 vs 7.3 ms at m16384)
_PPROG = {}
_BINDS = {}


def _binds(kind, rows):
    """TMA maps bounded to the rows actually written (6m): tiles of the last expert read past them as zeros
    instead of past the end of the workspace.  One small host-side descriptor per token count, no compile."""
    key = (kind, rows)
    b = _BINDS.get(key)
    if b is None:
        box = 32 if kind == "swap" else MB
        b = (tk.TiledTensorMapBinding(dtype="bf16", global_shape=(HID, rows), global_strides=(HID * 2,),
                                      box_shape=(KB, box), swizzle_bytes=128),
             tk.TiledTensorMapBinding(dtype="bf16", global_shape=(INTER, rows), global_strides=(INTER * 2,),
                                      box_shape=(KB, box), swizzle_bytes=128))
        _BINDS[key] = b
    return b


def _pget_prog(kind, args):
    p = _PPROG.get(kind)
    if p is None:
        p = tk.compile(_prefill_swap_prog if kind == "swap" else _prefill_prog, *args, target="sm103a")
        _PPROG[kind] = p
    return p


def moe_prefill(x: torch.Tensor, ids: torch.Tensor, tw: torch.Tensor, w1: torch.Tensor, s1: torch.Tensor,
                w2: torch.Tensor, s2: torch.Tensor, meta: torch.Tensor, picks: torch.Tensor, tile_e: torch.Tensor,
                tile_c: torch.Tensor, xs: torch.Tensor, inter: torch.Tensor,
                output: torch.Tensor, m: int, limit: float, alpha: float, beta: float) -> None:
    kind = "swap" if m <= SWAP_MAX_M else "pair"
    bx, bi = _binds(kind, m * TOPK)
    xs_map = bx.bind(xs.data_ptr())
    in_map = bi.bind(inter.data_ptr())
    u = tk.compiler.KernelArgument
    args = (x, ids, tw, w1, s1, w2, s2, meta, picks, tile_e, tile_c, xs, inter, output, xs_map, in_map,
            u("u32", m), u("u32", m * TOPK), u("f32", float(limit)), u("f32", float(alpha)), u("f32", float(beta)),
            torch.cuda.current_stream().cuda_stream)
    _pget_prog(kind, args)(*args)


KERNELS = {"moe_decode": ("cnt", "rows", "act", "nact", "gu", "inter", "osum", "output", "xp"),
           "moe_prefill": ("meta", "picks", "tile_e", "tile_c", "xs", "inter", "output")}


def apply(
    self,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
    apply_router_weight_on_input: bool,
) -> None:
    m = topk_ids.size(0)
    if (m <= DECODE_MAX and expert_tokens_meta is None and expert_map is None and a1q_scale is None
            and w1.size(0) <= 512):
        x = hidden_states.view(-1, hidden_states.size(-1))
        nexp = w1.size(0)
        dev = x.device
        cnt = torch.empty(nexp, dtype=torch.int32, device=dev)
        rows = torch.empty(nexp * CAP, dtype=torch.int32, device=dev)
        act = torch.empty(nexp, dtype=torch.int32, device=dev)
        nact = torch.empty(1, dtype=torch.int32, device=dev)
        gu = torch.empty(m * TOPK, N1, dtype=torch.float32, device=dev)
        inter = torch.empty(m * TOPK * G2 * REC2, dtype=torch.int32, device=dev)
        xp = torch.empty(m * G1 * REC2, dtype=torch.int32, device=dev)
        osum = torch.empty(m, N2, dtype=torch.float32, device=dev)
        moe_decode(x, topk_ids, topk_weights, w1, self.quant_config.w1_scale, w2, self.quant_config.w2_scale,
                   cnt, rows, act, nact, gu, inter, osum, output, xp, m, nexp, _clamp_limit(self))
        return
    cfg = self.activation_config
    limit = cfg.clamp_limit if cfg is not None else None
    if (m >= PREFILL_MIN and expert_tokens_meta is None and expert_map is None and a1q_scale is None
            and not apply_router_weight_on_input and activation == _MoEActivation.SILU and limit is not None
            and topk_ids.size(1) == TOPK and w1.size(0) == E):
        x = hidden_states.view(-1, hidden_states.size(-1))
        dev = x.device
        npick = m * TOPK
        meta = torch.empty(4, dtype=torch.int32, device=dev)
        picks = torch.empty(npick + 256, dtype=torch.int32, device=dev)
        ntiles = (npick + 255) // 256 + E
        tile_e = torch.empty(ntiles, dtype=torch.int32, device=dev)
        tile_c = torch.empty(ntiles, dtype=torch.int32, device=dev)
        # Compact expert-sorted rows in the engine's workspaces (sizes from workspace_shapes(m)):
        # workspace2 holds the gathered rows, then the weighted down rows (6m x 5120 bf16 = down_output);
        # workspace13 holds the activated intermediate (6m x 2304 bf16 = quanted_down_input).
        xs = workspace2.view(torch.bfloat16)[:npick * HID].view(npick, HID)
        inter = workspace13.view(torch.bfloat16)[:npick * INTER].view(npick, INTER)
        # SILU with a clamp limit is Humming's clamped SiLU with alpha=1, beta=0 (get_humming_activation).
        moe_prefill(x, topk_ids, topk_weights, w1, self.quant_config.w1_scale, w2, self.quant_config.w2_scale,
                    meta, picks, tile_e, tile_c, xs, inter, output, m, float(limit), 1.0, 0.0)
        return
    _original_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                    expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
                    apply_router_weight_on_input)


def _original_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
                    global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
                    expert_tokens_meta, apply_router_weight_on_input):
    """The engine's own Humming path, for token counts where it beats the TKDSL kernels or they do not cover."""
    hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    buffers = self.prepare_buffers(workspace13, workspace2, topk_ids.size(0), topk_ids.size(1), activation)
    buffers["output"] = output
    moe_kwargs1, moe_kwargs2, scatter_idx = self.prepare_humming_moe_kwargs(
        topk_ids=topk_ids, expert_map=expert_map, expert_tokens_meta=expert_tokens_meta)
    inputs, input_scale, input_scale_2 = self.process_input(
        "w13", inputs=hidden_states, input_scale=a1q_scale,
        quanted_input=buffers.get("quanted_gate_up_input", None))
    self.humming_forward("w13", inputs=inputs, weight=w1, input_scale=input_scale,
                         input_scale_2=input_scale_2, outputs=buffers["gate_up_output"], **moe_kwargs1)
    inputs, input_scale, input_scale_2 = self.process_input(
        "w2", inputs=buffers["gate_up_output"], quanted_input=buffers["quanted_down_input"],
        activation=activation, scatter_idx=scatter_idx)
    self.humming_forward("w2", inputs=inputs, weight=w2, input_scale=input_scale,
                         input_scale_2=input_scale_2,
                         outputs=buffers["down_output"].view(-1, hidden_states.size(-1)), **moe_kwargs2)
    valid_tokens = None
    if expert_tokens_meta and expert_tokens_meta.psum_recv_per_rank is not None:
        valid_tokens = expert_tokens_meta.psum_recv_per_rank[-1:]
    moe_fused_mul_sum(inputs=buffers["down_output"].view(*topk_ids.shape, -1), topk_weights=topk_weights,
                      topk_ids=topk_ids, expert_map=expert_map, outputs=output, num_valid_tokens=valid_tokens)
