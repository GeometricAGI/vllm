# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allreduce + residual add + RMSNorm latency, captured in CUDA graphs.

Compares the fused one-shot and two-shot push kernels (per cluster size)
with FlashInfer's fused allreduce (what the allreduce fusion pass uses by
default), and with an unfused allreduce (push, custom 1-stage, NCCL) followed
by vLLM's fused_add_rms_norm. Run one process per GPU:

    python benchmarks/kernels/benchmark_push_allreduce_rmsnorm.py \
        --world-size 8 --hidden 6144 --push-blocks 36 64 128

Each --push-blocks value is a separate run, because the push grid size is
fixed for the lifetime of a process.
"""

import argparse
import json
import os
import statistics
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm import _custom_ops as ops
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    graph_capture,
    init_distributed_environment,
)

EPS = 1e-5


def time_graph(fn, inner: int, reps: int, device) -> float:
    """Median µs per fn() call over reps replays of a graph of inner calls."""
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    with graph_capture(device=device) as ctx:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=ctx.stream):
            for _ in range(inner):
                fn()
    torch.accelerator.synchronize()
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(reps):
        dist.barrier(group=get_tp_group().cpu_group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / inner)
    del graph
    return statistics.median(samples)


def check(ca, group, world_size, tokens, hidden, cluster, in_place, two_shot, device):
    """Compare a fused push kernel with an fp32 reference and across ranks."""
    dtype = torch.bfloat16
    torch.manual_seed(1000 * tokens + cluster + dist.get_rank())
    inp = torch.randn(tokens, hidden, dtype=dtype, device=device)
    inp[0, :32] = 0.0
    inp[0, 1:32:2] = -0.0  # packed +0/-0 words collide with the sentinel
    torch.manual_seed(tokens)  # residual and gamma match on all ranks
    residual = torch.randn_like(inp)
    gamma = torch.randn(hidden, dtype=dtype, device=device)
    inputs = [torch.empty_like(inp) for _ in range(world_size)]
    dist.all_gather(inputs, inp, group=group)
    reduced = sum(x.float() for x in inputs)
    if two_shot:
        # The owner rounds the sum before every rank adds the residual.
        reduced = reduced.to(dtype).float()
    z = reduced + residual.float()
    expected = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + EPS) * gamma.float()
    if in_place:
        norm_out, residual_out = inp, residual
    else:
        norm_out, residual_out = torch.empty_like(inp), torch.empty_like(inp)
    ops.push_all_reduce_rmsnorm(
        ca._ptr,
        inp,
        residual,
        gamma,
        norm_out,
        residual_out,
        EPS,
        0.0,
        two_shot,
        cluster,
    )
    torch.testing.assert_close(residual_out, z.to(dtype), atol=0, rtol=0)
    torch.testing.assert_close(norm_out, expected.to(dtype), atol=2e-2, rtol=2e-2)
    outs = [torch.empty_like(norm_out) for _ in range(world_size)]
    dist.all_gather(outs, norm_out, group=group)
    assert all(torch.equal(outs[0], o) for o in outs), "ranks disagree"
    # The unfused push kernel shares the scratch and epochs: interleave it.
    x = torch.randn(tokens, hidden, dtype=dtype, device=device)
    xs = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(xs, x, group=group)
    torch.testing.assert_close(
        ca.push_all_reduce(x, "sentinel"),
        sum(v.float() for v in xs).to(dtype),
        atol=0,
        rtol=0,
    )


def worker(rank: int, args, push_blocks: int, port: int, out_q) -> None:
    os.environ.update(
        VLLM_ALLREDUCE_PUSH_MODE="sentinel",
        VLLM_ALLREDUCE_PUSH_MAX_SIZE_KB=str(args.push_max_kb),
        VLLM_ALLREDUCE_PUSH_BLOCKS=str(push_blocks),
    )
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_distributed_environment(
        world_size=args.world_size,
        rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=rank,
    )
    ensure_model_parallel_initialized(args.world_size, 1)
    tp = get_tp_group()
    comm = tp.device_communicator
    ca = comm.ca_comm
    assert ca is not None and ca.push_max_size > 0, "push allreduce not enabled"

    from vllm.compilation.passes.fusion import allreduce_rms_fusion as arf

    if args.check:
        n = 0
        for tokens in args.tokens:
            if tokens * args.hidden * 2 > ca.push_max_size:
                continue
            for cluster in [0, 1, 2, 4, 8]:
                if cluster and push_blocks % cluster:
                    continue
                for in_place in (True, False):
                    for two_shot in (False, True):
                        check(
                            ca,
                            tp.device_group,
                            args.world_size,
                            tokens,
                            args.hidden,
                            cluster,
                            in_place,
                            two_shot,
                            device,
                        )
                        n += 1
        torch.accelerator.synchronize()
        if rank == 0:
            print(f"push_blocks={push_blocks}: {n} correctness checks passed")

    dtype = torch.bfloat16
    gamma = torch.randn(args.hidden, dtype=dtype, device=device)
    results = []
    for tokens in args.tokens:
        if tokens * args.hidden * 2 > ca.push_max_size:
            continue
        inp = torch.randn(tokens, args.hidden, dtype=dtype, device=device)
        residual = torch.randn_like(inp)
        row = {"tokens": tokens, "kib": tokens * args.hidden * 2 / 1024}

        # The in-place form, as the fused-add pattern calls it (norm_out=None).
        def flashinfer_fused():
            arf.call_trtllm_fused_allreduce_norm(
                allreduce_in=inp,
                residual=residual,
                rms_gamma=gamma,
                rms_eps=EPS,
                world_size=args.world_size,
                launch_with_pdl=True,
                fp32_acc=True,
                max_token_num=args.fi_max_tokens,
                pattern_code=arf.ar_fusion_patterns.kARResidualRMSNorm,
            )

        with mock.patch.object(ca, "should_push_rmsnorm", return_value=False):
            row["flashinfer_fused"] = time_graph(
                flashinfer_fused, args.inner, args.reps, device
            )

        for two_shot in (False, True):
            for cluster in [0, 1, 2, 4, 8]:
                if cluster and push_blocks % cluster:
                    continue
                name = f"push_{'2shot' if two_shot else '1shot'}_c{cluster or 'auto'}"
                row[name] = time_graph(
                    lambda: ops.push_all_reduce_rmsnorm(
                        ca._ptr,
                        inp,
                        residual,
                        gamma,
                        inp,
                        residual,
                        EPS,
                        0.0,
                        two_shot,
                        cluster,
                    ),
                    args.inner,
                    args.reps,
                    device,
                )

        def unfused(all_reduce):
            def fn():
                out = all_reduce(inp)
                ops.fused_add_rms_norm(out, residual, gamma, EPS)

            return fn

        row["push+norm"] = time_graph(
            unfused(lambda x: ca.push_all_reduce(x, "sentinel")),
            args.inner,
            args.reps,
            device,
        )
        row["custom1stage+norm"] = time_graph(
            unfused(lambda x: ca.custom_all_reduce(x)), args.inner, args.reps, device
        )
        if comm.pynccl_comm is not None:
            row["nccl+norm"] = time_graph(
                unfused(lambda x: comm.pynccl_comm.all_reduce(x)),
                args.inner,
                args.reps,
                device,
            )
        # The slowest rank bounds every collective.
        gathered = [None] * args.world_size
        dist.all_gather_object(gathered, row, group=tp.cpu_group)
        if rank == 0:
            merged = {
                k: (max(r[k] for r in gathered) if isinstance(v, float) else v)
                for k, v in row.items()
            }
            merged["push_blocks"] = push_blocks
            results.append(merged)
    if rank == 0:
        out_q.put(results)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 48, 64, 85, 128, 170],
    )
    parser.add_argument("--push-blocks", type=int, nargs="+", default=[36])
    parser.add_argument("--push-max-kb", type=int, default=2048)
    parser.add_argument("--fi-max-tokens", type=int, default=2048)
    parser.add_argument("--inner", type=int, default=50)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--json", help="also write results to this file")
    parser.add_argument(
        "--check", action="store_true", help="check correctness before timing"
    )
    args = parser.parse_args()

    ctx = mp.get_context("spawn")
    all_results = []
    for i, push_blocks in enumerate(args.push_blocks):
        q = ctx.Queue()
        procs = [
            ctx.Process(target=worker, args=(r, args, push_blocks, 29500 + i, q))
            for r in range(args.world_size)
        ]
        for p in procs:
            p.start()
        all_results += q.get()
        for p in procs:
            p.join()
            assert p.exitcode == 0, f"worker exited with {p.exitcode}"

    keys = [k for k in all_results[0] if k not in ("tokens", "kib", "push_blocks")]
    print("µs per call, max over ranks, bf16 [tokens, hidden] in CUDA graphs")
    print(
        f"{'blocks':>6} {'tokens':>6} {'KiB':>6} " + " ".join(f"{k:>18}" for k in keys)
    )
    for r in all_results:
        print(
            f"{r['push_blocks']:>6} {r['tokens']:>6} {r['kib']:>6.0f} "
            + " ".join(f"{r.get(k, float('nan')):>18.2f}" for k in keys)
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
