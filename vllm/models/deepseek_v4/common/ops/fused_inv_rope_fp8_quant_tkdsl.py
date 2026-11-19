"""Fused inverse RoPE + UE8M0 block-scaled FP8 quantization in pure TKDSL.

Kernel (kernel-engineer-1, v3): a warp handles HPW heads of one token; lane l
owns elements 16l..16l+15 of each 512-wide head (two 16-byte bf16 loads, one
16-byte fp8 store), all loads issued before compute and pos/cos/sin shared.
RoPE pairs are adjacent, so a partner never leaves its lane; lanes 28..31 hold
the 64 rope dims. A 128-element quant block is 8 lanes: absmax by three xor
shuffles, UE8M0 word gathered to lane 0 by three index shuffles. Numerics
replicate the Triton baseline's PTX (fma contraction, libdevice log2f + ceil,
power-of-two divide, satfinite e4m3).

Host path (kernel-engineer-2): the first call per shape compiles/loads through
tk.compile; later calls allocate the outputs directly in their returned
strided layouts and launch the cached CUfunction through cuLaunchKernelEx with
a prebuilt config (PDL attribute kept) and a retained ABI argument block.
"""

import ctypes as _ct

import torch

from tkdsl import nvidia as tk

HEAD_DIM = 512
NOPE = 448
FP8_MAX = 448.0
EPS = 1e-10
INV_FP8_MAX = 1.0 / 448.0

_LOG2_COEFFS = (0xBE2C7F30, 0x3E2FCF2A, 0xBE374E43, 0x3E520BF4, 0xBE763C8B,
                0x3E93BF99, 0xBEB8AA49, 0x3EF6384A, 0xBF38AA3B)


def _f(bits):
    import struct
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _schedule(T):
    # (heads per warp, register cap); measured on B200 with the harness.
    # Small T: max parallelism at full occupancy.  Large T: 4 heads per warp
    # keeps 8 x 16B loads in flight per lane and amortizes per-token setup.
    if T >= 512:
        return 4, 72
    return 1, 32


def _build(T, T_al, H, G, hpg):
    HPW, MAXREG = _schedule(T)
    if hpg % HPW:
        HPW, MAXREG = 1, 32
    threads = 32 * hpg // HPW

    @tk.inline
    def rotate_pair(a, b, c, s):
        # Triton: even = fma(a, c, b*s); odd = fma(b, c, -(a*s))
        ra = tk.ops.fma(a, c, b * s)
        rb = tk.ops.fma(b, c, tk.ops.neg(a * s))
        return ra, rb

    @tk.inline
    def biased_ceil_log2(x):
        # libdevice __nv_log2f for normal x, then ceil; returns ceil + 127.
        i = tk.ops.bitcast(x, "u32")
        e = (i - 0x3F2AAAAB) & 0xFF800000
        m = tk.ops.bits_to_f32(i - e) - 1.0
        k = tk.ops.shift_right(e, 23, dtype="s32")
        fe = tk.ops.u32_to_f32((k + 256) & 0x1FF) - 256.0
        p = _f(0x3DC6B27F)
        for c in _LOG2_COEFFS:
            p = tk.ops.fma(p, m, _f(c))
        t = m * p
        t = m * t
        r = tk.ops.fma(m, _f(0x3FB8AA3B), t)
        f = fe + r
        neg = f <= 0.0
        tn = tk.ops.f32_to_u32_rz(tk.ops.neg(f))
        tp = tk.ops.f32_to_u32_rz(f)
        bump = tk.ops.select(tk.ops.u32_to_f32(tp) < f, 1, 0)
        return tk.ops.select(neg, 127 - tn, 127 + tp + bump)

    @tk.inline
    def biased_ceil_log2_fast(x):
        # Exhaustively verified over [2^-45, 2^121): ceil(__nv_log2f(x)) equals
        # exponent + (mantissa != 0) except when 0 < mantissa <= 22; those
        # rare inputs take the exact libdevice emulation.
        i = tk.ops.bitcast(x, "u32")
        mant = i & 0x7FFFFF
        e = (i >> 23) + tk.ops.select(mant != 0, 1, 0)
        if mant - 1 < 31:
            e = biased_ceil_log2(x)
        return e

    @tk.inline
    def quant_store(ys, out_ptr, scale_ptr, t, g, hl, lane):
        m = tk.ops.max3_f32(ys[0], ys[1], ys[2], absolute=True)
        for idx in range(3, 15, 2):
            m = tk.ops.max3_f32(m, ys[idx], ys[idx + 1], absolute=True)
        m = tk.ops.maximum(m, tk.ops.abs(ys[15]))
        m = tk.ops.maximum(m, tk.warp.shuffle_xor(m, 1))
        m = tk.ops.maximum(m, tk.warp.shuffle_xor(m, 2))
        m = tk.ops.maximum(m, tk.warp.shuffle_xor(m, 4))
        m = tk.ops.maximum(m, EPS)
        e = biased_ceil_log2_fast(m * INV_FP8_MAX)
        inv = tk.ops.bits_to_f32((254 - e) << 23)
        inv2 = tk.ops.pack((inv, inv), dtype=tk.F32)
        qs = []
        for k in range(4):
            vals = []
            for i in range(2):
                pr = tk.ops.pack((ys[4 * k + 2 * i], ys[4 * k + 2 * i + 1]), dtype=tk.F32)
                lo, hi = tk.ops.unpack(tk.ops.mul(pr, inv2))
                vals.append(lo)
                vals.append(hi)
            qs.append(tk.ops.pack(tuple(vals), dtype=tk.E4M3))
        dst = tk.ops.mad_wide_u32((g * T + t) * (hpg * 32) + hl * 32 + lane, 16, out_ptr)
        tk.gmem.store(dst, tuple(qs), width=4)
        e1 = tk.warp.shuffle_idx(e, 8)
        e2 = tk.warp.shuffle_idx(e, 16)
        e3 = tk.warp.shuffle_idx(e, 24)
        packed = e | (e1 << 8) | (e2 << 16) | (e3 << 24)
        if lane == 0:
            tk.gmem.store(
                tk.ops.mad_wide_u32((g * hpg + hl) * T_al + t, 4, scale_ptr),
                packed, dtype="u32",
            )

    @tk.kernel(reqntid=(threads, 1, 1), max_registers=MAXREG)
    def kernel(out_ptr, scale_ptr, o_ptr, pos_ptr, cache_ptr):
        tk.pdl.arrive()
        tk.pdl.wait()
        bid = tk.arch.block_idx_x()
        t = bid // G
        g = bid - t * G
        w = tk.arch.thread_idx_x() // 32
        lane = tk.arch.lane_id()
        if t >= T:
            if lane == 0:
                for j in range(HPW):
                    tk.gmem.store(
                        tk.ops.mad_wide_u32((g * hpg + w * HPW + j) * T_al + t, 4, scale_ptr),
                        0, dtype="u32",
                    )
        else:
            row0 = t * H + g * hpg + w * HPW
            all_words = []
            for j in range(HPW):
                src = tk.ops.mad_wide_u32((row0 + j) * 32 + lane, 32, o_ptr)
                w0 = tk.gmem.load("u32", src, width=4)
                w1 = tk.gmem.load("u32", src, width=4, byte_offset=16)
                all_words.append(list(w0) + list(w1))
            is_rope = lane >= 28
            pos = tk.gmem.load("u32", tk.ops.mad_wide_u32(t, 8, pos_ptr))
            rl = lane & 3
            cs = tk.ops.mad_wide_u32(pos * 16 + rl * 2, 16, cache_ptr)
            c0 = tk.gmem.load("f32", cs, width=4, mask=is_rope, other=1.0)
            c1 = tk.gmem.load("f32", cs, width=4, byte_offset=16, mask=is_rope, other=1.0)
            s0 = tk.gmem.load("f32", cs, width=4, byte_offset=128, mask=is_rope, other=0.0)
            s1 = tk.gmem.load("f32", cs, width=4, byte_offset=144, mask=is_rope, other=0.0)
            cos_v = list(c0) + list(c1)
            sin_v = list(s0) + list(s1)
            for j in range(HPW):
                xs = []
                for word in all_words[j]:
                    lo, hi = tk.ops.unpack(tk.ops.bitcast(word, "bf16x2"))
                    xs.append(lo)
                    xs.append(hi)
                ys = []
                for p in range(8):
                    a = xs[2 * p]
                    b = xs[2 * p + 1]
                    ra, rb = rotate_pair(a, b, cos_v[p], sin_v[p])
                    ys.append(tk.ops.select(is_rope, ra, a))
                    ys.append(tk.ops.select(is_rope, rb, b))
                quant_store(ys, out_ptr, scale_ptr, t, g, w * HPW + j, lane)

    grid = T_al * G

    @tk.jit
    def launch(out, scale, o, pos, cache, stream):
        kernel(out, scale, o, pos, cache).launch(
            grid=grid, block=threads, stream=stream,
            programmatic_stream_serialization=True,
        )

    p = tk.compiler.KernelArgument.pointer(0)
    return tk.compile(launch, p, p, p, p, p, 0, target="sm100a")


_CACHE: dict = {}
_raw_stream = torch._C._cuda_getCurrentRawStream
_empty_strided = torch.empty_strided
_empty = torch.empty
_FP8 = torch.float8_e4m3fn
_I32 = torch.int32


class _LaunchAttr(_ct.Structure):
    _fields_ = [("id", _ct.c_uint32), ("pad", _ct.c_uint32), ("value", _ct.c_uint8 * 64)]


class _LaunchConfig(_ct.Structure):
    _fields_ = [
        ("gx", _ct.c_uint32), ("gy", _ct.c_uint32), ("gz", _ct.c_uint32),
        ("bx", _ct.c_uint32), ("by", _ct.c_uint32), ("bz", _ct.c_uint32),
        ("smem", _ct.c_uint32), ("stream", _ct.c_void_p),
        ("attrs", _ct.POINTER(_LaunchAttr)), ("nattrs", _ct.c_uint32),
    ]


_libcuda = _ct.CDLL("libcuda.so.1")
_cuLaunchKernelEx = _libcuda.cuLaunchKernelEx
_cuLaunchKernelEx.argtypes = (_ct.c_void_p, _ct.c_void_p, _ct.c_void_p, _ct.c_void_p)
_cuLaunchKernelEx.restype = _ct.c_int


def _make_runner(compiled, T, n_groups, heads_per_group, device):
    """Bind a TKDSL-compiled launch once; return a per-call closure that only
    allocates fresh outputs, writes 5 pointers into the ABI block and calls
    cuLaunchKernelEx with a prebuilt config (PDL attribute included)."""
    (launch,) = compiled.launches
    T_al = (T + 3) // 4 * 4
    d = heads_per_group * HEAD_DIM
    inner = heads_per_group
    slots = (_ct.c_uint64 * 5)()
    params = (_ct.c_void_p * 5)(*(_ct.addressof(slots) + 8 * i for i in range(5)))
    attr = (_LaunchAttr * 1)()
    attr[0].id = 5  # CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    attr[0].value[0] = 1 if launch.programmatic_stream_serialization else 0
    cfg = _LaunchConfig()
    cfg.gx, cfg.gy, cfg.gz = launch.grid
    cfg.bx, cfg.by, cfg.bz = launch.block
    cfg.smem = launch.shared_memory
    cfg.stream = None
    cfg.attrs = attr
    cfg.nattrs = 1
    keep = (compiled, slots, params, attr, cfg)
    cfg_ref = _ct.byref(cfg)
    func = _ct.c_void_p(int(launch.function.handle))
    params_ref = _ct.c_void_p(_ct.addressof(params))
    oshape, ostride = (T, n_groups, d), (d, T * d, 1)
    sshape, sstride = (T, n_groups, inner), (1, inner * T_al, T_al)
    ssize = n_groups * inner * T_al
    exact = T % 4 == 0
    state = [None]
    launch_ex = _cuLaunchKernelEx
    raw_stream = _raw_stream
    es = _empty_strided
    em = _empty
    fp8 = _FP8
    i32 = _I32
    dindex = device.index

    def run(o, positions, cos_sin_cache):
        out = es(oshape, ostride, dtype=fp8, device=device)
        if exact:
            scale = es(sshape, sstride, dtype=i32, device=device)
        else:
            scale = em(ssize, dtype=i32, device=device).as_strided(sshape, sstride)
        stream = raw_stream(dindex)
        if stream != state[0]:
            cfg.stream = stream
            state[0] = stream
        slots[0] = out.data_ptr()
        slots[1] = scale.data_ptr()
        slots[2] = o.data_ptr()
        slots[3] = positions.data_ptr()
        slots[4] = cos_sin_cache.data_ptr()
        if launch_ex(cfg_ref, func, params_ref, None):
            raise RuntimeError("cuLaunchKernelEx failed")
        return out, scale

    run.keep = keep
    return run


def fused_inv_rope_fp8_quant(o: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor, n_groups: int, heads_per_group: int, nope_dim: int=448, rope_dim: int=64, quant_group_size: int=128, tma_aligned_scales: bool=False, quantize: bool=True):
    run = _CACHE.get((o.shape[0], n_groups, heads_per_group, o.get_device()))
    if run is None:
        T, H, hd = o.shape
        assert hd == HEAD_DIM and nope_dim == NOPE and rope_dim == 64
        assert quant_group_size == 128 and tma_aligned_scales and quantize
        assert H == n_groups * heads_per_group
        assert o.is_contiguous() and cos_sin_cache.is_contiguous()
        T_al = (T + 3) // 4 * 4
        d = heads_per_group * HEAD_DIM
        inner = heads_per_group
        prog = _build(T, T_al, H, n_groups, heads_per_group)
        # First call through TKDSL's own launcher (loads and binds the module).
        out = _empty((n_groups, T, d), dtype=_FP8, device=o.device)
        scale = _empty(n_groups * inner * T_al, dtype=_I32, device=o.device).as_strided(
            (n_groups, T, inner), (inner * T_al, 1, T_al))
        prog(out, scale, o, positions, cos_sin_cache, _raw_stream(o.get_device()))
        _CACHE[(T, n_groups, heads_per_group, o.get_device())] = _make_runner(
            prog, T, n_groups, heads_per_group, o.device)
        return out.transpose(0, 1), scale.transpose(0, 1)
    return run(o, positions, cos_sin_cache)
