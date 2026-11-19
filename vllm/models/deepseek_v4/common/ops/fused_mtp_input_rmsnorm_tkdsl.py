from __future__ import annotations

import torch

from tkdsl import nvidia as tk

HIDDEN = 4096
VECTOR = 8
LANES = HIDDEN // VECTOR  # 512
WARPS = LANES // 32


import os

CLUSTER = {1: int(os.environ.get("MTP_C1", "1")), 64: int(os.environ.get("MTP_C64", "1"))}
REDUX = os.environ.get("MTP_REDUX", "1") == "1"


def _compile(tokens: int, hc_mult: int, eps: float):
    cluster = CLUSTER.get(tokens, 1)
    fragment_spec = tk.vec.bf(VECTOR)
    e_spec = tk.gl("bf16", (tokens, HIDDEN))
    h_spec = tk.gl("bf16", (tokens * hc_mult, HIDDEN))
    w_spec = tk.gl("bf16", (HIDDEN,))
    scratch_spec = tk.tiles.sv("f32", WARPS)
    tasks = hc_mult + 1

    @tk.inline
    def norm_row(src, dst, weight_t, row, column, scratch, scale_mask):
        x = tk.load(fragment_spec, src, (row, column))
        w = tk.load(fragment_spec, weight_t, (0, column))
        local = tk.sum_squares([x])
        total = tk.sum(
            local,
            scratch=scratch,
            count=WARPS,
            width=WARPS,
            base=0,
            policy=tk.SumPolicy.LOCAL_SERIAL_VECTOR,
        )
        scale = tk.ops.rsqrt(tk.ops.fma(total, 1.0 / HIDDEN, eps), ftz=True)
        tk.store(dst, (x * w).scale(scale * scale_mask), (row, column))

    @tk.kernel(reqntid=(LANES, 1, 1))
    def kernel(e_ptr, pos_ptr, h_ptr, ew_ptr, hw_ptr, eo_ptr, ho_ptr):
        tid = tk.arch.thread_idx_x()
        column = tid * VECTOR
        block = tk.arch.block_idx_x()
        token = block // tasks
        task = block - token * tasks
        scratch = tk.shared_allocator().allocate(scratch_spec)
        ew = tk.tensor(w_spec, ew_ptr)
        hw = tk.tensor(w_spec, hw_ptr)
        if block >= ctas:
            return
        if task == 0:
            paddr = pos_ptr + tk.ops.mul_wide_u32(token, 8)
            lo = tk.gmem.load("u32", paddr)
            hi = tk.gmem.load("u32", paddr + tk.ops.mul_wide_u32(token * 0 + 1, 4))
            keep = 1.0 if (lo | hi) != 0 else 0.0
            norm_row(tk.tensor(e_spec, e_ptr), tk.tensor(e_spec, eo_ptr), ew, token, column, scratch, keep)
        else:
            norm_row(
                tk.tensor(h_spec, h_ptr),
                tk.tensor(h_spec, ho_ptr),
                hw,
                token * hc_mult + task - 1,
                column,
                scratch,
                1.0,
            )

    ctas = tokens * tasks
    grid = ((ctas + cluster - 1) // cluster) * cluster

    @tk.jit
    def launch(e, pos, h, ew, hw, eo, ho, stream):
        if cluster == 1:
            kernel(e, pos, h, ew, hw, eo, ho).launch(grid=grid, block=LANES, stream=stream)
        else:
            kernel(e, pos, h, ew, hw, eo, ho).launch(grid=grid, block=LANES, cluster=cluster, stream=stream)

    pointer = tk.compiler.KernelArgument.pointer(0)
    target = tk.current_specialization(required=True).target
    if REDUX:
        target = tk.profile(target).with_plans(warp_sum_f32="redux_fixed")
    return tk.compile(launch, pointer, pointer, pointer, pointer, pointer, pointer, pointer, 0, target=target)


@tk.specialize(key=lambda e, pos, h, ew, hw, eps, hc: (e.shape[0], int(hc), float(eps)))
def _compiled(tokens, hc_mult, eps):
    return _compile(tokens, hc_mult, eps)


@torch.no_grad()
def fused_mtp_input_rmsnorm(inputs_embeds, positions, previous_hidden_states, enorm_weight, hnorm_weight, eps, hc_mult):
    tokens = inputs_embeds.shape[0]
    enorm_out = torch.empty_like(inputs_embeds)
    hnorm_out = torch.empty_like(previous_hidden_states)
    if tokens == 0:
        return enorm_out, hnorm_out
    program = _compiled(inputs_embeds, positions, previous_hidden_states, enorm_weight, hnorm_weight, eps, hc_mult)
    program(
        inputs_embeds, positions, previous_hidden_states, enorm_weight, hnorm_weight,
        enorm_out, hnorm_out, torch.cuda.current_stream(inputs_embeds.device),
    )
    return enorm_out, hnorm_out
