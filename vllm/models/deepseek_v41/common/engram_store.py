# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One copy of each CPU-offloaded Engram table for every server on a host.

Each server process normally pins a private ~94 GiB copy of every Engram
table. With ``EngramConfig.shared_store`` set, servers that point at the same
directory and load the same checkpoint map a single copy instead. The copy
lives in a memfd, so its size is bounded by host memory rather than by
``/dev/shm``, and it is freed when the last server holding it exits.

A registry file per table in the store directory names the processes that
hold the memfd. A server attaches by opening ``/proc/<pid>/fd/<fd>`` of any
live holder; when none is alive it creates a new memfd. Weights are written
by whichever server loads them first. A later server compares a fingerprint
of the checkpoint shard with the one recorded and skips the copy when they
match, or refuses to start when they differ.
"""

import fcntl
import hashlib
import json
import mmap
import os
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)

_FINGERPRINT_SAMPLES = 64
_FINGERPRINT_BLOCK = 1 << 16


def _start_time(pid: int) -> str | None:
    """The kernel start time of ``pid``, which tells a live holder from a
    reused pid."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat.rsplit(")", 1)[1].split()[19]


def fingerprint(shard: torch.Tensor) -> str:
    """Hash of a checkpoint shard's shape, dtype and sampled bytes."""
    data = shard.contiguous().view(torch.uint8).reshape(-1).numpy()
    digest = hashlib.sha256(f"{tuple(shard.shape)}:{shard.dtype}".encode())
    step = max(1, (data.size - _FINGERPRINT_BLOCK) // _FINGERPRINT_SAMPLES)
    for start in range(0, max(1, data.size - _FINGERPRINT_BLOCK + 1), step):
        digest.update(data[start : start + _FINGERPRINT_BLOCK].tobytes())
    digest.update(data[-_FINGERPRINT_BLOCK:].tobytes())
    return digest.hexdigest()


def register_host_memory(owner: np.ndarray, mapping: mmap.mmap) -> torch.Tensor:
    """Pin a shared mapping for UVA access and tie its lifetime to the tensor."""
    tensor = torch.from_numpy(owner)
    pointer = tensor.data_ptr()
    result = torch.cuda.cudart().cudaHostRegister(pointer, owner.nbytes, 0)
    if result.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {result}")

    def unregister(mapping: mmap.mmap) -> None:
        # Holding the mapping keeps it mapped until CUDA releases it.
        if torch.cuda.cudart().cudaHostUnregister(pointer).value != 0:
            logger.warning("Engram store cudaHostUnregister failed")

    finalizer = weakref.finalize(owner, unregister, mapping)
    finalizer.atexit = False  # type: ignore[misc]
    if not tensor.is_pinned():
        raise RuntimeError("cudaHostRegister did not pin the Engram store mapping")
    return tensor


class NodeSharedEngramStorage:
    """Host weights of one Engram table shared by every server on the host.

    Args:
        directory: The store directory, shared by the servers that share tables.
        identity: What makes two tables the same (model, revision, layer, shard
            bounds, shape); servers attach only to a table with equal identity.
        num_rows: Rows of this table shard.
        dim: Row width in FP8 bytes.
        block_size: Columns per E8M0 scale byte.
        register: Pins the mapping; replaceable where CUDA is unavailable.

    """

    def __init__(
        self,
        directory: str,
        identity: dict[str, Any],
        num_rows: int,
        dim: int,
        block_size: int,
        register: Callable[[np.ndarray, mmap.mmap], torch.Tensor] = (
            register_host_memory
        ),
    ) -> None:
        weight_bytes = num_rows * dim
        self.nbytes = weight_bytes + weight_bytes // block_size
        self.identity = dict(identity, nbytes=self.nbytes)
        key = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode()
        ).hexdigest()[:24]
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._registry = self.directory / f"{key}.json"
        self._lock = self.directory / f"{key}.lock"
        self._fd, self.attached = self._open(key)
        mapping = mmap.mmap(self._fd, self.nbytes, flags=mmap.MAP_SHARED)
        storage = register(np.frombuffer(mapping, dtype=np.uint8), mapping)
        self.weight = storage[:weight_bytes].view(torch.float8_e4m3fn).view(-1, dim)
        self.weight_scale_inv = storage[weight_bytes:].view(-1, dim // block_size)
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None
        self.copied: list[str] = []

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        """The registry, read and rewritten under an exclusive lock."""
        with open(self._lock, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                registry = json.loads(self._registry.read_text())
            except (OSError, ValueError):
                registry = {}
            yield registry
            tmp = self._registry.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(registry, indent=1))
            tmp.replace(self._registry)

    def _open(self, key: str) -> tuple[int, bool]:
        """Attach to a live holder's memfd, or create the table's memfd."""
        with self._locked() as registry:
            if registry and registry.get("identity") != self.identity:
                raise RuntimeError(
                    f"Engram store {self._registry} holds a different table"
                )
            holders = [
                h
                for h in registry.get("holders", [])
                if h["pid"] != os.getpid() and _start_time(h["pid"]) == h["start"]
            ]
            fd = None
            for holder in holders:
                try:
                    fd = os.open(
                        f"/proc/{holder['pid']}/fd/{holder['fd']}",
                        os.O_RDWR | os.O_CLOEXEC,
                    )
                except OSError:
                    continue
                if os.fstat(fd).st_size == self.nbytes:
                    break
                os.close(fd)
                fd = None
            attached = fd is not None
            if fd is None:
                fd = os.memfd_create(f"vllm_engram_{key}", os.MFD_CLOEXEC)
                os.ftruncate(fd, self.nbytes)
                registry.clear()
                registry.update(identity=self.identity, loaded={})
            # This process keeps the fd open so later servers can reach it.
            holders.append(
                {"pid": os.getpid(), "fd": fd, "start": _start_time(os.getpid())}
            )
            registry["holders"] = holders
        logger.info(
            "Engram store %s: %s %.2f GiB table",
            self._registry.name,
            "attached to the shared" if attached else "created a shared",
            self.nbytes / 1024**3,
        )
        return fd, attached

    def load_weight(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        """Copy this table's shard unless another server already wrote it."""
        from vllm.models.deepseek_v41.common.engram import (
            _engram_head_shard_weight_loader,
        )

        name = (
            "weight"
            if param.data_ptr() == self.weight.data_ptr()
            else "weight_scale_inv"
        )
        if loaded_weight.dtype == torch.float8_e8m0fnu:
            loaded_weight = loaded_weight.view(torch.uint8)
        shard = loaded_weight.narrow(0, param.engram_vocab_start, param.shape[0])
        expected = fingerprint(shard)
        with self._locked() as registry:
            recorded = registry.get("loaded", {}).get(name)
        if recorded == expected:
            logger.info(
                "Engram store %s: %s already loaded by another server, not copied",
                self._registry.name,
                name,
            )
            return
        if recorded is not None:
            raise RuntimeError(
                f"Engram store {self._registry} was loaded from a different "
                f"checkpoint ({name} fingerprint differs)"
            )
        # Every writer copies the same bytes, so a concurrent copy is harmless.
        _engram_head_shard_weight_loader(param, loaded_weight)
        self.copied.append(name)
        with self._locked() as registry:
            registry.setdefault("loaded", {})[name] = expected

    def get_views(
        self, weight: torch.Tensor, scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (weight.data_ptr(), scales.data_ptr()) != (
            self.weight.data_ptr(),
            self.weight_scale_inv.data_ptr(),
        ):
            raise RuntimeError("Shared Engram parameter storage must not be replaced")
        if self._views is None:
            self._views = (
                get_accelerator_view_from_cpu_tensor(self.weight),
                get_accelerator_view_from_cpu_tensor(self.weight_scale_inv),
            )
        return self._views
