from __future__ import annotations

import torch
from tkdsl import nvidia as tk

HIDDEN = 4096
VEC = 8
LANES = 256
FRAGS = HIDDEN // (VEC * LANES)
STRIDE = VEC * LANES


def _compile(T: int, hc: int, eps: float):
    frag = tk.vec.bf(VEC)
    warps = LANES // 32
    rows_h = T * hc
    emb_spec = tk.gl("bf16", (T, HIDDEN))
    prev_spec = tk.gl("bf16", (rows_h, HIDDEN))
    w_spec = tk.gl("bf16", (HIDDEN,))
    pos_spec = tk.gl("s32", (2 * T,))
    scratch_spec = tk.tiles.sv("f32", 32)

    def load_frags(tensor, row, column, predicate=None):
        vals = []
        for f in range(FRAGS):
            col = column if f == 0 else tk.ops.add(column, f * STRIDE, dtype='u32')
            vals.append(tk.load(frag, tensor, (row, col), vector_words=4, predicate=predicate))
        return vals

    @tk.inline
    def do_row(src, dst, wt, row, column, lane, warp, scratch, keep):
        def norm_frag(v, w, scale):
            out = []
            for a, b in zip(v.values, w.values):
                a0, a1 = tk.ops.unpack(a)
                b0, b1 = tk.ops.unpack(b)
                out.append(tk.ops.pack((a0 * scale * b0, a1 * scale * b1), dtype="bf16"))
            return v._new(tuple(out))

        values = load_frags(src, row, column)
        weights = load_frags(wt, 0, column)
        ss = tk.sum_squares(values, policy=tk.vec.ReductionPolicy.ACCURATE)
        total = tk.sum(
            ss, scratch=scratch, index=warp, count=warps, width=warps,
            lane=lane, base=0, policy=tk.SumPolicy.LOCAL_SERIAL_VECTOR,
        )
        scale = tk.ops.rsqrt(total * (1.0 / HIDDEN) + eps, ftz=True)
        scale = scale if keep else 0.0
        for f in range(FRAGS):
            col = column if f == 0 else tk.ops.add(column, f * STRIDE, dtype='u32')
            tk.store(dst, norm_frag(values[f], weights[f], scale), (row, col), vector_words=4)

    @tk.kernel(reqntid=(LANES, 1, 1))
    def kernel(eout_ptr, hout_ptr, emb_ptr, prev_ptr, pos_ptr, ew_ptr, hw_ptr):
        tid = tk.arch.thread_idx_x()
        lane = tk.arch.lane_id()
        warp = tid // 32
        column = tid * VEC
        b = tk.arch.block_idx_x()
        eout = tk.tensor(emb_spec, eout_ptr)
        hout = tk.tensor(prev_spec, hout_ptr)
        emb = tk.tensor(emb_spec, emb_ptr)
        prev = tk.tensor(prev_spec, prev_ptr)
        pos = tk.tensor(pos_spec, pos_ptr)
        ew = tk.tensor(w_spec, ew_ptr)
        hw = tk.tensor(w_spec, hw_ptr)
        scratch = tk.shared_allocator().allocate(scratch_spec)
        if b < T:
            lo = tk.load(pos, (b * 2,), mask=b < T, other=0)
            hi = tk.load(pos, (b * 2 + 1,), mask=b < T, other=0)
            keep = (lo | hi) != 0
            do_row(emb, eout, ew, b, column, lane, warp, scratch, keep)
        else:
            do_row(prev, hout, hw, b - T, column, lane, warp, scratch, True)

    @tk.jit
    def launch(eout, hout, emb, prev, pos, ew, hw, stream):
        kernel(eout, hout, emb, prev, pos, ew, hw).launch(
            grid=T + rows_h, block=LANES, stream=stream)

    p = tk.compiler.KernelArgument.pointer(0)
    return tk.compile(launch, p, p, p, p, p, p, p, 0)


@tk.specialize(key=lambda T, hc, eps: (T, hc, eps))
def _compiled(T, hc, eps):
    return _compile(T, hc, eps)


def fused_mtp_input_rmsnorm(inputs_embeds, positions, previous_hidden_states, enorm_weight, hnorm_weight, *, eps, hc_mult):
    T, H = inputs_embeds.shape
    assert H == HIDDEN
    eout = torch.empty_like(inputs_embeds)
    hout = torch.empty_like(previous_hidden_states)
    prog = _compiled(T, hc_mult, float(eps))
    prog(eout, hout, inputs_embeds, previous_hidden_states, positions.view(torch.int32),
         enorm_weight, hnorm_weight, torch.cuda.current_stream(inputs_embeds.device))
    return eout, hout
