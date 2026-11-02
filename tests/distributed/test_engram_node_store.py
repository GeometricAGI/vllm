# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate servers on one host map one copy of each Engram table."""

import multiprocessing
import sys

import numpy as np
import pytest
import torch

from vllm.config.engram import EngramConfig
from vllm.models.deepseek_v41.common.engram_store import NodeSharedEngramStorage

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="memfd and /proc are Linux-only"
)

ROWS, DIM, BLOCK = 64, 32, 32
IDENTITY = {"model": "m", "revision": "r", "layer": "layers.1.engram"}


def unpinned(owner: np.ndarray, mapping) -> torch.Tensor:
    return torch.from_numpy(owner)


def store(directory: str) -> NodeSharedEngramStorage:
    return NodeSharedEngramStorage(
        directory, IDENTITY, ROWS, DIM, BLOCK, register=unpinned
    )


def checkpoint(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 255, (ROWS, DIM), dtype=torch.uint8, generator=generator)
    scales = torch.randint(
        0, 255, (ROWS, DIM // BLOCK), dtype=torch.uint8, generator=generator
    )
    return weight.view(torch.float8_e4m3fn), scales


def load(storage: NodeSharedEngramStorage, seed: int) -> None:
    for tensor, loaded in zip(
        (storage.weight, storage.weight_scale_inv), checkpoint(seed)
    ):
        param = torch.nn.Parameter(tensor, requires_grad=False)
        param.engram_vocab_start = 0
        storage.load_weight(param, loaded)


def first_server(directory: str, loaded, done) -> None:
    storage = store(directory)
    load(storage, seed=0)
    loaded.set()
    done.wait(30)


@pytest.fixture
def running_server(tmp_path):
    context = multiprocessing.get_context("spawn")
    loaded, done = context.Event(), context.Event()
    process = context.Process(target=first_server, args=(str(tmp_path), loaded, done))
    process.start()
    assert loaded.wait(60)
    yield
    done.set()
    process.join(30)


def test_a_second_server_maps_the_loaded_table_without_copying(
    tmp_path, running_server
):
    storage = store(str(tmp_path))
    load(storage, seed=0)
    assert storage.attached
    assert storage.copied == []
    weight, scales = checkpoint(0)
    assert torch.equal(storage.weight.view(torch.uint8), weight.view(torch.uint8))
    assert torch.equal(storage.weight_scale_inv, scales)


def test_a_different_checkpoint_is_refused(tmp_path, running_server):
    storage = store(str(tmp_path))
    with pytest.raises(RuntimeError, match="different checkpoint"):
        load(storage, seed=1)


def test_a_table_whose_holders_all_exited_is_created_and_loaded_again(tmp_path):
    context = multiprocessing.get_context("spawn")
    loaded, done = context.Event(), context.Event()
    done.set()
    process = context.Process(target=first_server, args=(str(tmp_path), loaded, done))
    process.start()
    process.join(60)
    storage = store(str(tmp_path))
    load(storage, seed=0)
    assert not storage.attached
    assert storage.copied == ["weight", "weight_scale_inv"]


def test_shared_store_excludes_the_other_host_storage_modes():
    for options in (
        {"use_thp": True},
        {"dp_shared_memory": True},
        {"cpu_offload": False},
    ):
        with pytest.raises(ValueError, match="shared_store requires"):
            EngramConfig(shared_store="/dev/shm/engram", **options)


def test_shared_store_does_not_change_the_compilation_hash():
    assert (
        EngramConfig(shared_store="/dev/shm/engram").compute_hash()
        == EngramConfig().compute_hash()
    )
