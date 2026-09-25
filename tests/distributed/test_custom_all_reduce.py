# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import random

import pytest
import ray
import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce  # noqa
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.parallel_state import get_tp_group, graph_capture

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

random.seed(42)
test_sizes = [random.randint(1024, 2048 * 1024) for _ in range(8)]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float32, True),
        (torch.float16, True),
        (torch.bfloat16, True),
        (torch.int8, False),
        (torch.float8_e4m3fn, False),
    ],
)
def test_custom_allreduce_filters_dtype(
    dtype: torch.dtype,
    expected: bool,
) -> None:
    communicator = car.CustomAllreduce.__new__(car.CustomAllreduce)
    communicator.disabled = False
    communicator.world_size = 2
    communicator.max_size = 1024
    communicator._ptr = 0

    assert communicator.should_custom_ar(torch.empty(16, dtype=dtype)) is expected


@pytest.mark.parametrize(
    ("major", "local_multicast", "expected"),
    [
        (8, True, False),
        (9, True, False),
        (10, False, False),
        (10, True, True),
    ],
)
def test_cross_node_mnnvl_gate_checks_generation_and_multicast(
    monkeypatch,
    major,
    local_multicast,
    expected,
):
    def has_device_capability(capability, device_id):
        assert capability == 100
        assert device_id == 3
        return major >= 10

    monkeypatch.setattr(
        car.current_platform,
        "has_device_capability",
        has_device_capability,
    )
    monkeypatch.setattr(
        car,
        "_has_local_multicast_support",
        lambda _device: local_multicast,
    )
    monkeypatch.setattr(car.dist, "all_reduce", lambda *_args, **_kwargs: None)

    assert car._group_can_attempt_mnnvl(object(), torch.device("cuda:3")) is expected


def test_cross_node_mnnvl_gate_requires_support_on_every_rank(monkeypatch):
    monkeypatch.setattr(
        car.current_platform,
        "has_device_capability",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        car,
        "_has_local_multicast_support",
        lambda _device: True,
    )

    def report_unsupported_peer(support, **_kwargs):
        support.zero_()

    monkeypatch.setattr(car.dist, "all_reduce", report_unsupported_peer)

    assert not car._group_can_attempt_mnnvl(object(), torch.device("cuda:0"))


def test_local_multicast_support_rejects_non_cuda(monkeypatch):
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: False)

    assert not car._has_local_multicast_support(torch.device("cuda:0"))


@ray.remote(num_gpus=1, max_calls=1)
def graph_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group

        # A small all_reduce for warmup.
        # this is needed because device communicators might be created lazily
        # (e.g. NCCL). This will ensure that the communicator is initialized
        # before any communication happens, so that this group can be used for
        # graph capture immediately.
        data = torch.zeros(1)
        data = data.to(device=device)
        torch.distributed.all_reduce(data, group=group)
        torch.accelerator.synchronize()
        del data

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1

        for sz in test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                with graph_capture(device=device) as graph_capture_context:
                    # use integers so result matches NCCL exactly
                    device_idx = torch.accelerator.current_device_index()
                    inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)
                    inp2 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)

                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                        for i in range(num_communication):
                            out1 = tensor_model_parallel_all_reduce(inp1)
                            # the input buffer is immediately modified to test
                            # synchronization
                            dist.all_reduce(inp1, group=group)
                            out2 = tensor_model_parallel_all_reduce(inp2)
                            dist.all_reduce(inp2, group=group)
                graph.replay()
                torch.testing.assert_close(out1, inp1)
                torch.testing.assert_close(out2, inp2)


@ray.remote(num_gpus=1, max_calls=1)
def eager_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1
        sz = 1024
        fa = get_tp_group().device_communicator.ca_comm
        inp = torch.ones(sz, dtype=torch.float32, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))

        inp = torch.ones(sz * 4, dtype=torch.bfloat16, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))


@ray.remote(num_gpus=1, max_calls=1)
def push_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        mode = os.environ["VLLM_ALLREDUCE_PUSH_MODE"]
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group
        fa = get_tp_group().device_communicator.ca_comm
        assert fa is not None and fa.push_max_size == 1024 * 1024

        def check(inp, out):
            # Exact reference: fp32 sum in rank order, like the kernel.
            inputs = [torch.empty_like(inp) for _ in range(tp_size)]
            dist.all_gather(inputs, inp, group=group)
            expected = sum(x.float() for x in inputs).to(inp.dtype)
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
            outputs = [torch.empty_like(out) for _ in range(tp_size)]
            dist.all_gather(outputs, out, group=group)
            assert all(torch.equal(outputs[0], o) for o in outputs)

        def make_input(n, dtype, seed):
            torch.manual_seed(seed * tp_size + rank)
            inp = torch.randn(n, dtype=dtype, device=device)
            # Packed +0/-0 words collide with the sentinel.
            inp[:32] = 0.0
            inp[1:32:2] = -0.0
            return inp

        # Interleave sizes, dtypes and the 1-stage kernel to exercise the
        # double-buffered scratch and the device-side epochs.
        sizes = [8, 4096, 8 * 4096 + 64, 256 * 1024, 512 * 1024]
        for seed, dtype in enumerate([torch.float32, torch.float16, torch.bfloat16]):
            for n in random.Random(seed).sample(sizes, len(sizes)):
                inp = make_input(n, dtype, seed)
                if inp.nbytes > fa.push_max_size:
                    continue
                assert fa.push_sync_mode(inp) == mode
                check(inp, tensor_model_parallel_all_reduce(inp))
                fa.all_reduce(inp, registered=False)

        inps = [make_input(n, torch.bfloat16, 0) for n in sizes[:-1]]
        with graph_capture(device=device) as graph_capture_context:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                outs = [tensor_model_parallel_all_reduce(inp) for inp in inps]
        for step in range(10):
            for i, inp in enumerate(inps):
                inp.copy_(make_input(inp.numel(), inp.dtype, step * 10 + i))
            graph.replay()
            for inp, out in zip(inps, outs):
                check(inp, out)


def push_allreduce_rmsnorm(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group
        fa = get_tp_group().device_communicator.ca_comm
        assert fa is not None and fa.push_max_size == 1024 * 1024
        eps = 1e-5

        def make(rows, hidden, dtype, seed):
            torch.manual_seed(seed * tp_size + rank)
            inp = torch.randn(rows, hidden, dtype=dtype, device=device)
            # Packed +0/-0 words collide with the sentinel.
            inp[0, :32] = 0.0
            inp[0, 1:32:2] = -0.0
            # Residual and gamma must match on all ranks, as in a model.
            torch.manual_seed(seed)
            residual = torch.randn(rows, hidden, dtype=dtype, device=device)
            gamma = torch.randn(hidden, dtype=dtype, device=device)
            return inp, residual, gamma

        def check(inp, residual, gamma, norm_out, residual_out, weight_bias):
            inputs = [torch.empty_like(inp) for _ in range(tp_size)]
            dist.all_gather(inputs, inp, group=group)
            z = sum(x.float() for x in inputs) + residual.float()
            # Exact: fp32 sum in rank order plus the residual, like the kernel.
            torch.testing.assert_close(residual_out, z.to(inp.dtype), atol=0, rtol=0)
            expected = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps)
            expected = expected * (gamma.float() + weight_bias)
            torch.testing.assert_close(
                norm_out, expected.to(inp.dtype), atol=2e-2, rtol=2e-2
            )
            outputs = [torch.empty_like(norm_out) for _ in range(tp_size)]
            dist.all_gather(outputs, norm_out, group=group)
            assert all(torch.equal(outputs[0], o) for o in outputs)

        # Interleave shapes, cluster sizes, in-place and out-of-place calls
        # and the unfused push kernel, which shares the scratch and epochs.
        shapes = [(1, 6144), (4, 6144), (37, 6144), (85, 6144), (3, 4096)]
        rng = random.Random(0)
        for seed, dtype in enumerate([torch.float16, torch.bfloat16] * 3):
            for rows, hidden in rng.sample(shapes, len(shapes)):
                inp, residual, gamma = make(rows, hidden, dtype, seed)
                assert fa.should_push_rmsnorm(inp, gamma)
                inp_ref, residual_ref = inp.clone(), residual.clone()
                weight_bias = rng.choice([0.0, 1.0])
                cluster_size = rng.choice([0, 1, 2, 4, 8])
                if rng.random() < 0.5:
                    norm_out, residual_out = inp, residual
                else:
                    norm_out = torch.empty_like(inp)
                    residual_out = torch.empty_like(inp)
                ops.push_all_reduce_rmsnorm(
                    fa._ptr,
                    inp,
                    residual,
                    gamma,
                    norm_out,
                    residual_out,
                    eps,
                    weight_bias,
                    cluster_size,
                )
                check(inp_ref, residual_ref, gamma, norm_out, residual_out, weight_bias)
                out = tensor_model_parallel_all_reduce(inp_ref)
                torch.testing.assert_close(
                    out, sum_over_ranks(inp_ref, group, tp_size), atol=0, rtol=0
                )

        inp, residual, gamma = make(4, 6144, torch.bfloat16, 0)
        norm_out = torch.empty_like(inp)
        with graph_capture(device=device) as graph_capture_context:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                fa.push_all_reduce_rmsnorm(inp, residual, gamma, norm_out, inp, eps)
        for step in range(10):
            new_inp, new_residual, _ = make(4, 6144, torch.bfloat16, step)
            inp.copy_(new_inp)
            residual.copy_(new_residual)
            graph.replay()
            check(new_inp, new_residual, gamma, norm_out, inp, 0.0)


def sum_over_ranks(inp, group, tp_size):
    inputs = [torch.empty_like(inp) for _ in range(tp_size)]
    dist.all_gather(inputs, inp, group=group)
    return sum(x.float() for x in inputs).to(inp.dtype)


@pytest.mark.parametrize("tp_size", [2, 8])
def test_push_allreduce_rmsnorm(monkeypatch: pytest.MonkeyPatch, tp_size):
    if torch.accelerator.device_count() < tp_size:
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ALLREDUCE_PUSH_MODE", "sentinel")
    monkeypatch.setenv("VLLM_ALLREDUCE_PUSH_MAX_SIZE_KB", "1024")
    multi_process_parallel(monkeypatch, tp_size, 1, push_allreduce_rmsnorm)


@pytest.mark.parametrize("mode", ["ll", "sentinel"])
def test_push_allreduce(monkeypatch: pytest.MonkeyPatch, mode):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ALLREDUCE_PUSH_MODE", mode)
    monkeypatch.setenv("VLLM_ALLREDUCE_PUSH_MAX_SIZE_KB", "1024")
    multi_process_parallel(monkeypatch, 2, 1, push_allreduce)


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("test_target", [eager_allreduce, graph_allreduce])
def test_custom_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pipeline_parallel_size,
    test_target,
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, pipeline_parallel_size, test_target)
