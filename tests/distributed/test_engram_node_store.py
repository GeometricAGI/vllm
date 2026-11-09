# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate servers on one host map one copy of each Engram table."""

import gc
import json
import mmap
import multiprocessing
import os
import signal
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.config.engram import EngramConfig
from vllm.models.deepseek_v41.common import engram_store
from vllm.models.deepseek_v41.common.engram_store import NodeSharedEngramStorage

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="memfd and /proc are Linux-only"
)

ROWS, DIM, BLOCK = 64, 32, 32
IDENTITY = {"model": "m", "revision": "r", "layer": "layers.1.engram"}


def unpinned(owner: np.ndarray, mapping) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # The shared mapping is read-only.
        return torch.from_numpy(owner)


def store(directory: str) -> NodeSharedEngramStorage:
    return NodeSharedEngramStorage(
        directory, IDENTITY, ROWS, DIM, BLOCK, register=unpinned
    )


def registry_path(directory: str) -> Path:
    (path,) = Path(directory).glob("*.json")
    return path


def checkpoint(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 255, (ROWS, DIM), dtype=torch.uint8, generator=generator)
    scales = torch.randint(
        0, 255, (ROWS, DIM // BLOCK), dtype=torch.uint8, generator=generator
    )
    return weight.view(torch.float8_e4m3fn), scales


def load(storage: NodeSharedEngramStorage, weights) -> None:
    for tensor, loaded in zip((storage.weight, storage.weight_scale_inv), weights):
        param = torch.nn.Parameter(tensor, requires_grad=False)
        param.engram_vocab_start = 0
        storage.load_weight(param, loaded)


def holds(storage: NodeSharedEngramStorage, weights) -> bool:
    weight, scales = weights
    return torch.equal(
        storage.weight.view(torch.uint8), weight.view(torch.uint8)
    ) and torch.equal(storage.weight_scale_inv, scales)


def first_server(directory: str, loaded, done) -> None:
    storage = store(directory)
    load(storage, checkpoint(0))
    loaded.set()
    done.wait(30)


@pytest.fixture
def running_server(tmp_path):
    context = multiprocessing.get_context("spawn")
    loaded, done = context.Event(), context.Event()
    process = context.Process(target=first_server, args=(str(tmp_path), loaded, done))
    process.start()
    assert loaded.wait(60)
    yield process
    done.set()
    process.join(30)


def test_a_second_server_maps_the_loaded_table_without_copying(
    tmp_path, running_server
):
    registry = registry_path(str(tmp_path))
    storage = store(str(tmp_path))
    rewritten = registry.stat().st_ino
    load(storage, checkpoint(0))
    assert storage.attached
    assert storage.copied == []
    assert holds(storage, checkpoint(0))
    # Finding the table loaded leaves the registry as it was.
    assert registry.stat().st_ino == rewritten


def test_a_different_checkpoint_is_refused(tmp_path, running_server):
    storage = store(str(tmp_path))
    with pytest.raises(RuntimeError, match="different checkpoint"):
        load(storage, checkpoint(1))


def test_a_checkpoint_differing_in_one_byte_is_refused(tmp_path, running_server):
    weight, scales = checkpoint(0)
    weight = weight.view(torch.uint8).clone()
    weight[ROWS // 3, DIM // 3] ^= 1
    storage = store(str(tmp_path))
    with pytest.raises(RuntimeError, match="weight differs"):
        load(storage, (weight.view(torch.float8_e4m3fn), scales))
    assert holds(storage, checkpoint(0))


def racing_server(directory: str, seed: int, start, results, done) -> None:
    storage = store(directory)
    start.wait(30)
    try:
        load(storage, checkpoint(seed))
    except RuntimeError as exc:
        results.put((seed, str(exc)))
    else:
        results.put((seed, holds(storage, checkpoint(seed))))
    done.wait(30)


def test_servers_loading_different_checkpoints_at_once_never_mix_them(tmp_path):
    context = multiprocessing.get_context("spawn")
    start, done, results = context.Barrier(3), context.Event(), context.Queue()
    processes = [
        context.Process(
            target=racing_server, args=(str(tmp_path), seed, start, results, done)
        )
        for seed in (0, 1)
    ]
    for process in processes:
        process.start()
    start.wait(60)
    outcomes = dict(results.get(timeout=60) for _ in processes)
    done.set()
    for process in processes:
        process.join(30)
    loaded = [seed for seed, outcome in outcomes.items() if outcome is True]
    refused = [
        seed
        for seed, outcome in outcomes.items()
        if isinstance(outcome, str) and "different checkpoint" in outcome
    ]
    assert len(loaded) == 1 and len(refused) == 1, outcomes


def test_a_table_whose_holders_all_exited_is_created_and_loaded_again(tmp_path):
    context = multiprocessing.get_context("spawn")
    loaded, done = context.Event(), context.Event()
    done.set()
    process = context.Process(target=first_server, args=(str(tmp_path), loaded, done))
    process.start()
    process.join(60)
    storage = store(str(tmp_path))
    load(storage, checkpoint(1))
    assert not storage.attached
    assert storage.copied == ["weight", "weight_scale_inv"]


def test_a_live_holder_that_cannot_be_opened_is_not_duplicated(
    tmp_path, running_server, monkeypatch
):
    def denied(path, *args):
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr(engram_store.os, "open", denied)
    with pytest.raises(RuntimeError, match="cannot open"):
        store(str(tmp_path))
    assert len(json.loads(registry_path(str(tmp_path)).read_text())["holders"]) == 1


def test_a_holder_in_another_pid_namespace_is_not_duplicated(tmp_path, running_server):
    path = registry_path(str(tmp_path))
    registry = json.loads(path.read_text())
    registry["holders"][0]["pid_ns"] = "pid:[1]"
    path.write_text(json.dumps(registry))
    with pytest.raises(RuntimeError, match="PID namespace"):
        store(str(tmp_path))


def test_a_holder_fd_naming_another_file_is_skipped(tmp_path, running_server):
    path = registry_path(str(tmp_path))
    registry = json.loads(path.read_text())
    registry["holders"][0]["fd"] = 0  # stdin, not the table's memfd.
    path.write_text(json.dumps(registry))
    storage = store(str(tmp_path))
    assert not storage.attached
    holders = json.loads(path.read_text())["holders"]
    assert [holder["pid"] for holder in holders] == [os.getpid()]


@pytest.mark.parametrize("release", ["close", "collect"])
def test_a_dropped_store_releases_its_table(tmp_path, release):
    storage = store(str(tmp_path))
    load(storage, checkpoint(0))
    fd = storage._fd
    if release == "close":
        storage.close()
    else:
        del storage
        gc.collect()
    with pytest.raises(OSError):
        os.fstat(fd)
    assert json.loads(registry_path(str(tmp_path)).read_text())["holders"] == []
    again = store(str(tmp_path))
    load(again, checkpoint(1))
    assert not again.attached and holds(again, checkpoint(1))


def write_outside_load_weight(directory: str) -> None:
    storage = store(directory)
    load(storage, checkpoint(0))
    storage.weight.view(torch.uint8).fill_(0)


def test_the_table_cannot_be_written_outside_load_weight(tmp_path):
    process = multiprocessing.get_context("spawn").Process(
        target=write_outside_load_weight, args=(str(tmp_path),)
    )
    process.start()
    process.join(60)
    assert process.exitcode == -signal.SIGSEGV


def permissions(address: int) -> str:
    """The protection of the mapping holding ``address``, as /proc/self/maps
    gives it (``r--s``, ``rw-s``)."""
    for line in Path("/proc/self/maps").read_text().splitlines():
        span, perms = line.split()[:2]
        start, end = (int(bound, 16) for bound in span.split("-"))
        if start <= address < end:
            return perms
    raise AssertionError(f"{address:#x} is not mapped")


class TestReadOnlyRegistration:
    """How the store pins its read-only mapping, with or without the device's
    read-only registration."""

    @pytest.fixture
    def table(self):
        fd = os.memfd_create("engram_test", os.MFD_CLOEXEC)
        os.ftruncate(fd, 1 << 20)
        mapping = mmap.mmap(fd, 1 << 20, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
        yield np.frombuffer(mapping, dtype=np.uint8), mapping
        os.close(fd)

    def register(self, seen: list):
        def fake(owner, mapping, flags=0):
            seen.append((flags, permissions(owner.ctypes.data)))
            return unpinned(owner, mapping)

        return fake

    def test_a_device_with_read_only_registration_registers_it_read_only(self, table):
        owner, mapping = table
        seen: list = []
        engram_store.register_read_only(
            owner, mapping, supported=lambda: True, register=self.register(seen)
        )
        assert seen == [(engram_store._CUDA_HOST_REGISTER_READ_ONLY, "r--s")]
        assert permissions(owner.ctypes.data) == "r--s"

    def test_a_device_without_it_registers_it_writable_and_maps_it_read_only(
        self, table
    ):
        owner, mapping = table
        seen: list = []
        engram_store.register_read_only(
            owner, mapping, supported=lambda: False, register=self.register(seen)
        )
        assert seen == [(0, "rw-s")]
        assert permissions(owner.ctypes.data) == "r--s"

    def test_a_failed_registration_leaves_it_read_only(self, table):
        owner, mapping = table

        def refuse(owner, mapping, flags=0):
            raise RuntimeError("cudaHostRegister failed")

        with pytest.raises(RuntimeError, match="cudaHostRegister failed"):
            engram_store.register_read_only(
                owner, mapping, supported=lambda: False, register=refuse
            )
        assert permissions(owner.ctypes.data) == "r--s"


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


@pytest.mark.parametrize("load_format", ["dummy", "sharded_state", "tensorizer"])
def test_loaders_that_bypass_load_weight_are_refused(load_format):
    from types import SimpleNamespace

    from vllm.models.deepseek_v41.common.engram import Engram

    config = SimpleNamespace(load_config=SimpleNamespace(load_format=load_format))
    with pytest.raises(ValueError, match="cannot be used with load_format"):
        Engram._store_identity(config, "layers.1.engram.embed_tokens")


def test_the_store_identity_names_the_checkpoint_layer_and_tp_shard(monkeypatch):
    from types import SimpleNamespace

    from vllm.models.deepseek_v41.common import engram as engram_ops

    monkeypatch.setattr(engram_ops, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(engram_ops, "get_tensor_model_parallel_world_size", lambda: 2)
    config = SimpleNamespace(
        load_config=SimpleNamespace(load_format="auto"),
        model_config=SimpleNamespace(model="m", revision="r"),
    )
    assert engram_ops.Engram._store_identity(config, "layer") == {
        "model": "m",
        "revision": "r",
        "layer": "layer",
        "tp": [1, 2],
    }
