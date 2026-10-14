from __future__ import annotations

import torch

from tkdsl import nvidia as tk

HIDDEN = 4096
VEC = 16
LANES = HIDDEN // VEC  # 512 threads
WARPS = LANES // 32


def _build(T, hc, eps):
    spec = tk.vec.bf(VEC)
    e_spec = tk.gl("bf16", (T, HIDDEN))
    h_spec = tk.gl("bf16", (T * hc, HIDDEN))
    w_spec = tk.gl("bf16", (HIDDEN,))
    p_spec = tk.gl("s32", (2 * T,))
    scratch_spec = tk.tiles.sv("f32", WARPS)
    tasks = hc + 1
    inv_h = 1.0 / HIDDEN

    @tk.kernel(reqntid=(LANES, 1, 1))
    def kernel(eout_ptr, hout_ptr, e_ptr, h_ptr, ew_ptr, hw_ptr, pos_ptr):
        tid = tk.arch.thread_idx_x()
        column = tid * VEC
        scratch = tk.shared_allocator().allocate(scratch_spec)
        bid = tk.arch.block_idx_x()
        token = bid // tasks
        task = bid - token * tasks
        if task == 0:
            eo = tk.tensor(e_spec, eout_ptr)
            ei = tk.tensor(e_spec, e_ptr)
            ew = tk.tensor(w_spec, ew_ptr)
            positions = tk.tensor(p_spec, pos_ptr)
            plo = positions[token * 2,]
            phi = positions[token * 2 + 1,]
            nonzero = (plo | phi) != 0
            x = tk.load(spec, ei, (token, column), vector_words=8)
            w = tk.load(spec, ew, (column,), vector_words=8)
            s = tk.sum(tk.sum_squares([x]), scratch=scratch, count=WARPS, width=WARPS, base=0)
            scale = tk.ops.rsqrt(tk.ops.fma(s, inv_h, eps), ftz=True)
            scale = scale if nonzero else 0.0
            tk.store(eo, (x * w).scale(scale), (token, column), vector_words=8)
        else:
            ho = tk.tensor(h_spec, hout_ptr)
            hi = tk.tensor(h_spec, h_ptr)
            hw = tk.tensor(w_spec, hw_ptr)
            row = token * hc + (task - 1)
            x = tk.load(spec, hi, (row, column), vector_words=8)
            w = tk.load(spec, hw, (column,), vector_words=8)
            s = tk.sum(tk.sum_squares([x]), scratch=scratch, count=WARPS, width=WARPS, base=0)
            scale = tk.ops.rsqrt(tk.ops.fma(s, inv_h, eps), ftz=True)
            tk.store(ho, (x * w).scale(scale), (row, column), vector_words=8)

    @tk.jit
    def launch(eout, hout, e, h, ew, hw, pos, stream):
        kernel(eout, hout, e, h, ew, hw, pos).launch(grid=T * tasks, block=LANES, stream=stream)

    pointer = tk.compiler.KernelArgument.pointer(0)
    return tk.compile(launch, pointer, pointer, pointer, pointer, pointer, pointer, pointer, 0)


_CACHE = {}


def fused_mtp_input_rmsnorm(inputs_embeds, positions, previous_hidden_states, enorm_weight, hnorm_weight, eps, hc_mult):
    T = inputs_embeds.shape[0]
    enorm_out = torch.empty_like(inputs_embeds)
    hnorm_out = torch.empty_like(previous_hidden_states)
    key = (T, hc_mult, float(eps))
    prog = _CACHE.get(key)
    if prog is None:
        prog = _CACHE[key] = _build(T, hc_mult, float(eps))
    prog(enorm_out, hnorm_out, inputs_embeds, previous_hidden_states, enorm_weight, hnorm_weight,
         positions.view(torch.int32),
         torch.cuda.current_stream())
    return enorm_out, hnorm_out
