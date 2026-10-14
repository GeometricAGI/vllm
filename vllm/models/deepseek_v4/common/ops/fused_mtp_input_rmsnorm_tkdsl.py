from __future__ import annotations

import torch

from tkdsl import nvidia as tk

HIDDEN = 4096
EPS = 1e-6
VECTOR = 8


def _compile(T: int, hc: int, lanes: int, cluster: int, policy, red):
    fragments = HIDDEN // (VECTOR * lanes)
    warps = max(1, lanes // 32)
    width = 1 << (warps - 1).bit_length()
    fspec = tk.vec.bf(VECTOR)
    e_spec = tk.gl("bf16", (T, HIDDEN))
    h_spec = tk.gl("bf16", (T * hc, HIDDEN))
    w_spec = tk.gl("bf16", (HIDDEN,))
    p_spec = tk.gl("u32", (2 * T,))
    scratch_spec = tk.tiles.sv("f32", 32)
    stride = lanes * VECTOR
    in_cache = tk.gmem.CachePolicy(l1="evict_first", l2="evict_first", fraction=1.0)
    w_cache = tk.gmem.CachePolicy(l1="evict_last", l2="evict_last", fraction=1.0)

    def load_frags(tensor, row, column, cache=tk.gmem.CachePolicy.DEFAULT):
        vals = []
        for f in range(fragments):
            col = column if f == 0 else tk.ops.add(column, f * stride, dtype="u32")
            vals.append(tk.load(fspec, tensor, (row, col), cache=cache))
        return vals

    if warps == 1:
        def reduce_row(value, lane, tid, scratch):
            return tk.warp.sum(value, width=lanes)
    else:
        @tk.inline
        def reduce_row(value, lane, tid, scratch):
            return tk.sum(value, scratch=scratch, index=tid // 32, count=warps,
                          width=width, lane=lane, base=0, policy=policy)

    @tk.inline
    def process(src, wt, out, row, column, lane, tid, scratch, scale_mask):
        vals = load_frags(src, row, column, in_cache)
        ws = load_frags(wt, 0, column, w_cache)
        s = reduce_row(tk.sum_squares(vals, policy=red), lane, tid, scratch)
        scale = tk.ops.rsqrt(s * (1.0 / HIDDEN) + EPS, ftz=True)
        if scale_mask is not None:
            scale = tk.ops.select(scale_mask, scale, 0.0)
        for f in range(fragments):
            col = column if f == 0 else tk.ops.add(column, f * stride, dtype="u32")
            words = []
            for xw, ww in zip(vals[f].values, ws[f].values, strict=True):
                x0, x1 = tk.ops.unpack(xw, dtype=tk.BF16)
                w0, w1 = tk.ops.unpack(ww, dtype=tk.BF16)
                y0 = tk.ops.mul(tk.ops.mul(x0, scale), w0)
                y1 = tk.ops.mul(tk.ops.mul(x1, scale), w1)
                words.append(tk.ops.pack((y0, y1), dtype="bf16"))
            tk.gmem.store(out.element_addr((row, col)), value=tuple(words), dtype="bf16x2")

    @tk.kernel(reqntid=(lanes, 1, 1))
    def kernel(eo_ptr, ho_ptr, emb_ptr, prev_ptr, ew_ptr, hw_ptr, pos_ptr):
        tid = tk.arch.thread_idx_x()
        lane = tk.arch.lane_id()
        column = tid * VECTOR
        eo = tk.tensor(e_spec, eo_ptr)
        ho = tk.tensor(h_spec, ho_ptr)
        emb = tk.tensor(e_spec, emb_ptr)
        prev = tk.tensor(h_spec, prev_ptr)
        ew = tk.tensor(w_spec, ew_ptr)
        hw = tk.tensor(w_spec, hw_ptr)
        pos = tk.tensor(p_spec, pos_ptr)
        scratch = None if warps == 1 else tk.shared_allocator().allocate(scratch_spec)
        b = tk.arch.block_idx_x()
        if b < T:
            lo = tk.load(pos, (2 * b,), mask=True, other=0)
            hi = tk.load(pos, (2 * b + 1,), mask=True, other=0)
            keep = (lo | hi) != 0
            process(emb, ew, eo, b, column, lane, tid, scratch, keep)
        else:
            process(prev, hw, ho, b - T, column, lane, tid, scratch, None)

    @tk.jit
    def launch(eo, ho, emb, prev, ew, hw, pos, stream):
        kernel(eo, ho, emb, prev, ew, hw, pos).launch(
            grid=T * (hc + 1), block=lanes, cluster=cluster, stream=stream)

    p = tk.compiler.KernelArgument.pointer(0)
    return tk.compile(launch, p, p, p, p, p, p, p, 0)


def _cfg(T):
    return {1: (512, 1), 64: (512, 1), 512: (128, 1)}.get(T, (256, 1))


@tk.specialize(key=lambda emb, pos, prev, ew, hw: (emb.shape[0],))
def _compiled(T):
    lanes, cluster = _cfg(T)
    return _compile(T, 4, lanes, cluster, tk.SumPolicy.LOCAL_SERIAL_VECTOR,
                    tk.vec.ReductionPolicy.ACCURATE)


@torch.no_grad()
def fused_mtp_input_rmsnorm(inputs_embeds, positions, previous_hidden_states,
                            enorm_weight, hnorm_weight, eps, hc_mult):
    T = inputs_embeds.shape[0]
    eo = torch.empty_like(inputs_embeds)
    ho = torch.empty_like(previous_hidden_states)
    if T == 0:
        return eo, ho
    prog = _compiled(inputs_embeds, positions, previous_hidden_states, enorm_weight, hnorm_weight)
    prog(eo, ho, inputs_embeds, previous_hidden_states.view(T * hc_mult, HIDDEN),
         enorm_weight, hnorm_weight, positions.view(torch.int32),
         torch.cuda.current_stream(inputs_embeds.device))
    return eo, ho
