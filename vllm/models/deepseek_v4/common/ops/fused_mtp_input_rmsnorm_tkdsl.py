from dataclasses import dataclass

import torch

from tkdsl import nvidia as tk

H = 4096
HC = 4
EPS = 1e-6


@dataclass(frozen=True)
class _Sched:
    vector: int = 8
    lanes: int = 512
    cross_warp: tk.SumPolicy = tk.SumPolicy.LOCAL_SERIAL_VECTOR
    reduction: tk.vec.ReductionPolicy = tk.vec.ReductionPolicy.PACKED
    redux: bool = False


def _schedule(T):
    if T <= 64:
        return _Sched(lanes=256)
    return _Sched(lanes=256, redux=False)


def _compile(T, s):
    vector, lanes = s.vector, s.lanes
    fragments = H // (vector * lanes)
    warps = max(1, lanes // 32)
    rw = 1 << (warps - 1).bit_length()
    stride = lanes * vector
    fspec = tk.vec.bf(vector)
    vw = 8 if vector == 16 else None
    emb_spec = tk.gl("bf16", (T, H))
    prev_spec = tk.gl("bf16", (T * HC, H))
    w_spec = tk.gl("bf16", (H,))
    scratch_spec = tk.tiles.sv("f32", rw)
    default = tk.gmem.CachePolicy.DEFAULT

    def load_frags(tensor, row, column, predicate=None, cp=default):
        vals = []
        for f in range(fragments):
            col = column if f == 0 else tk.ops.add(column, f * stride, dtype="u32")
            vals.append(tk.load(fspec, tensor, (row, col), cache=cp,
                                vector_words=vw, predicate=predicate))
        return vals

    @tk.inline
    def reduce_row(value, scratch):
        if warps == 1:
            return tk.warp.sum(value, width=lanes)
        return tk.sum(value, scratch=scratch, count=warps, width=rw, base=0,
                      policy=s.cross_warp)

    @tk.inline
    def process(src, wt, out, row, column, scratch, keep):
        vals = load_frags(src, row, column, keep, tk.gmem.CachePolicy.EVICT_FIRST)
        ws = load_frags(wt, 0, column, None, tk.gmem.CachePolicy.EVICT_LAST)
        row_sum = reduce_row(tk.sum_squares(vals, policy=s.reduction), scratch)
        scale = tk.ops.rsqrt(tk.ops.fma(row_sum, 1.0 / H, EPS), ftz=True)
        for f, (v, w) in enumerate(zip(vals, ws, strict=True)):
            col = column if f == 0 else tk.ops.add(column, f * stride, dtype="u32")
            packed = []
            for xv, wv in zip(v.values, w.values, strict=True):
                x0, x1 = tk.ops.unpack(xv)
                w0, w1 = tk.ops.unpack(wv)
                y0 = tk.ops.mul(tk.ops.mul(x0, scale), w0)
                y1 = tk.ops.mul(tk.ops.mul(x1, scale), w1)
                packed.append(tk.ops.pack((y0, y1), dtype="bf16"))
            res = type(v)(v.shape, tuple(packed))
            tk.store(out, res, (row, col), cache=default, vector_words=vw)

    @tk.kernel(reqntid=(lanes, 1, 1))
    def kernel(eout_ptr, hout_ptr, emb_ptr, prev_ptr, pos_ptr, ew_ptr, hw_ptr):
        tid = tk.arch.thread_idx_x()
        column = tid * vector
        t = tk.arch.block_idx_x()
        k = tk.arch.block_idx_y()
        scratch = None if warps == 1 else tk.shared_allocator().allocate(scratch_spec)
        if k == 0:
            lo = tk.gmem.load("u32", tk.ops.mad_wide_u32(t, 8, pos_ptr))
            hi = tk.gmem.load("u32", tk.ops.mad_wide_u32(t * 2 + 1, 4, pos_ptr))
            keep = (lo | hi) != 0
            process(tk.tensor(emb_spec, emb_ptr), tk.tensor(w_spec, ew_ptr),
                    tk.tensor(emb_spec, eout_ptr), t, column, scratch, keep)
        else:
            process(tk.tensor(prev_spec, prev_ptr), tk.tensor(w_spec, hw_ptr),
                    tk.tensor(prev_spec, hout_ptr), t * HC + (k - 1), column, scratch, None)

    @tk.jit
    def launch(eout, hout, emb, prev, pos, ew, hw, stream):
        kernel(eout, hout, emb, prev, pos, ew, hw).launch(
            grid=(T, HC + 1), block=lanes, stream=stream)

    ptr = tk.compiler.KernelArgument.pointer(0)
    ctx = tk.current_specialization(required=True)
    target = ctx.target
    if s.redux:
        target = tk.profile(ctx.target).with_plans(warp_sum_f32="redux_fixed")
    return tk.compile(launch, ptr, ptr, ptr, ptr, ptr, ptr, ptr, 0, target=target)


@tk.specialize(key=lambda emb, eout: (emb.shape[0], _schedule(emb.shape[0])))
def _compiled(T, sched):
    return _compile(T, sched)


@torch.no_grad()
def fused_mtp_input_rmsnorm(inputs_embeds, positions, previous_hidden_states, enorm_weight, hnorm_weight, eps, hc_mult):
    eout = torch.empty_like(inputs_embeds)
    hout = torch.empty_like(previous_hidden_states)
    prog = _compiled(inputs_embeds, eout)
    prog(eout, hout, inputs_embeds, previous_hidden_states, positions, enorm_weight, hnorm_weight,
         torch.cuda.current_stream(inputs_embeds.device))
    return eout, hout
