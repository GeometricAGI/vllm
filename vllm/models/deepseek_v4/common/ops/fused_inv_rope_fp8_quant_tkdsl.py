"""Pure-TKDSL port of vLLM's fused inverse-RoPE + UE8M0 block FP8 quantization."""

import struct

import torch

from tkdsl import nvidia as tk

HEAD_DIM = 512
QB = 128
EPS = 1e-10


def _f(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


INV448 = _f(0x3B124925)
_PC = tuple(_f(b) for b in (0x3DC6B27F, 0xBE2C7F30, 0x3E2FCF2A, 0xBE374E43, 0x3E520BF4,
                            0xBE763C8B, 0x3E93BF99, 0xBEB8AA49, 0x3EF6384A, 0xBF38AA3B))
_LOG2E = _f(0x3FB8AA3B)


def _build(T, H, G):
    HPG = H // G
    TP = (T + 3) // 4 * 4
    WARPS = 4
    threads = WARPS * 32
    tblocks = TP // WARPS
    ctas = tblocks * H
    D = HPG * HEAD_DIM
    inner = D // QB // 4

    @tk.kernel(reqntid=(threads, 1, 1))
    def kernel(out_ptr, o_ptr, pos_ptr, cache_ptr, scale_ptr):
        tid = tk.arch.thread_idx_x()
        lane = tid & 31
        warp = tid >> 5
        bid = tk.arch.block_idx_x()
        head = bid // tblocks
        t = (bid - head * tblocks) * WARPS + warp
        g = head // HPG
        hig = head - g * HPG
        saddr = tk.ops.mad_wide_u32(g * (inner * TP) + hig * TP + t, 4, scale_ptr)
        if t >= T:
            if lane == 0:
                tk.gmem.store(saddr, tk.ops.constant_i32(0), dtype="u32")
            return
        in_addr = tk.ops.mad_wide_u32((t * H + head) * HEAD_DIM + lane * 16, 2, o_ptr)
        w0 = tk.gmem.load("u32", in_addr, width=4)
        w1 = tk.gmem.load("u32", in_addr, width=4, byte_offset=16)
        words = list(w0) + list(w1)
        xs = []
        for w in words:
            xs.append(tk.ops.bitcast(w << 16, "f32"))
            xs.append(tk.ops.bitcast(w & 0xFFFF0000, "f32"))
        if lane >= 28:
            p = tk.gmem.load("u32", tk.ops.mad_wide_u32(t, 8, pos_ptr))
            cbase = tk.ops.mad_wide_u32(p * 64 + (lane - 28) * 8, 4, cache_ptr)
            c0 = tk.gmem.load("f32", cbase, width=4)
            c1 = tk.gmem.load("f32", cbase, width=4, byte_offset=16)
            s0 = tk.gmem.load("f32", cbase, width=4, byte_offset=128)
            s1 = tk.gmem.load("f32", cbase, width=4, byte_offset=144)
            cs = list(c0) + list(c1)
            ss = list(s0) + list(s1)
            ys = []
            for j in range(8):
                xe = xs[2 * j]
                xo = xs[2 * j + 1]
                ys.append(tk.ops.fma(xe, cs[j], tk.ops.mul(xo, ss[j])))
                ys.append(tk.ops.fma(xo, cs[j], tk.ops.neg(tk.ops.mul(xe, ss[j]))))
            r = ys
        else:
            r = xs
        m = tk.ops.abs(r[0])
        for v in r[1:]:
            m = tk.ops.maximum(m, tk.ops.abs(v))
        m = tk.ops.maximum(m, tk.warp.shuffle_xor(m, 4))
        m = tk.ops.maximum(m, tk.warp.shuffle_xor(m, 2))
        m = tk.ops.maximum(m, tk.warp.shuffle_xor(m, 1))
        m = tk.ops.maximum(m, EPS)
        s = tk.ops.mul(m, INV448)
        # __nv_log2f(s), normal positive path (s >= 1e-10/448 is normal).
        sb = tk.ops.bitcast(s, "u32")
        eb = (sb - 0x3F3504F3) & 0xFF800000
        mf = tk.ops.bitcast(sb - eb, "f32")
        kk = tk.ops.shift_right(tk.ops.bitcast(eb, "s32"), 23, dtype="s32")
        ku = tk.ops.bitcast(kk, "u32") + 128
        kf = tk.ops.sub(tk.ops.u32_to_f32(ku), 128.0)
        tt = tk.ops.add(mf, -1.0)
        q = tk.ops.fma(_PC[0], tt, _PC[1])
        for c in _PC[2:]:
            q = tk.ops.fma(q, tt, c)
        q = tk.ops.mul(tt, q)
        q = tk.ops.mul(tt, q)
        q = tk.ops.fma(tt, _LOG2E, q)
        L = tk.ops.add(kf, q)
        # ceil(L) exactly: L = fl(k + r) with |r| < 1.
        nu = tk.ops.select(tk.ops.gt(L, kf), ku + 1, ku)  # n + 128
        rcp = tk.ops.bitcast((255 - nu) << 23, "f32")  # 2^-n
        packed = []
        for i in range(4):
            vals = []
            for k in range(4):
                v = tk.ops.mul(r[4 * i + k], rcp)
                v = tk.ops.minimum(tk.ops.maximum(v, -448.0), 448.0)
                vals.append(v)
            packed.append(tk.ops.bitcast(tk.ops.pack(tuple(vals), dtype="e4m3"), "u32"))
        out_addr = tk.ops.mad_wide_u32((g * T + t) * D + hig * HEAD_DIM + lane * 16, 1, out_ptr)
        tk.gmem.store(out_addr, tuple(packed), width=4)
        sbyte = (nu - 1) << (lane & 24)
        sbyte = sbyte + tk.warp.shuffle_xor(sbyte, 16)
        sbyte = sbyte + tk.warp.shuffle_xor(sbyte, 8)
        if lane == 0:
            tk.gmem.store(saddr, sbyte, dtype="u32")

    @tk.jit
    def launch(out, o, pos, cache, scale, stream):
        kernel(out, o, pos, cache, scale).launch(grid=ctas, block=threads, stream=stream)

    pointer = tk.compiler.KernelArgument.pointer(0)
    return tk.compile(launch, pointer, pointer, pointer, pointer, pointer, 0, target="sm100a")


_PROGRAMS = {}


def fused_inv_rope_fp8_quant(o: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor, n_groups: int, heads_per_group: int, nope_dim: int=448, rope_dim: int=64, quant_group_size: int=128, tma_aligned_scales: bool=False, quantize: bool=True):
    T, H, _ = o.shape
    G = n_groups
    key = (T, H, G)
    prog = _PROGRAMS.get(key)
    if prog is None:
        prog = _PROGRAMS[key] = _build(T, H, G)
    TP = (T + 3) // 4 * 4
    D = heads_per_group * HEAD_DIM
    inner = D // QB // 4
    out_buf = torch.empty((G, T, D), dtype=torch.float8_e4m3fn, device=o.device)
    scale_buf = torch.empty(G * inner * TP, dtype=torch.int32, device=o.device).as_strided(
        (G, T, inner), (inner * TP, 1, TP))
    prog(out_buf, o, positions, cos_sin_cache, scale_buf, torch.cuda.current_stream(o.device))
    return out_buf.transpose(0, 1), scale_buf.transpose(0, 1)
