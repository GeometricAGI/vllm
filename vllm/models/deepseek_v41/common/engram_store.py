# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One copy of each CPU-offloaded Engram table for every server on a host.

Each server process normally pins a private ~94 GiB copy of every Engram
table. With ``EngramConfig.shared_store`` set, servers that point at the same
directory and load the same checkpoint map a single copy instead. The copy
lives in a memfd, so its size is bounded by host memory rather than by
``/dev/shm``, and it is freed when the last server holding it exits.

A registry file per table in the store directory names the memfd (by inode)
and the processes that hold it. A server attaches by opening
``/proc/<pid>/fd/<fd>`` of a live holder; when no holder is alive it creates
a new memfd, and when a live holder cannot be opened it refuses to start
rather than create a second copy.

Each table is written once, by whichever server loads it first, while that
server holds the table's lock. A later server compares its checkpoint shard
byte for byte with the table and skips the copy when they match, or refuses
to start when they differ. Servers map the table read-only, so only
``load_weight`` can write it; any other write faults instead of changing a
table another server is serving. A device that cannot register read-only
memory gets the mapping made writable for its registration only.
"""

import ctypes
import fcntl
import hashlib
import json
import mmap
import os
import warnings
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.logger import init_logger
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)

# cudaHostRegisterReadOnly: the device only reads the registered range, which
# CUDA requires to register a mapping without write access.
_CUDA_HOST_REGISTER_READ_ONLY = 0x08
# cudaDevAttrHostRegisterReadOnlySupported: whether the device takes that flag.
_CUDA_DEV_ATTR_HOST_REGISTER_READ_ONLY_SUPPORTED = 113

_libc = ctypes.CDLL(None, use_errno=True)


def engram_head_shard(
    param: torch.nn.Parameter, loaded_weight: torch.Tensor
) -> torch.Tensor:
    """This rank's complete head buckets of a checkpoint table. ue8m0 scales
    arrive as float8_e8m0fnu; keep the raw bytes (the param stores uint8)."""
    if loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)
    shard = loaded_weight.narrow(0, param.engram_vocab_start, param.shape[0])
    assert shard.shape == param.shape, (
        f"engram shard {tuple(shard.shape)} does not fit param {tuple(param.shape)}"
    )
    return shard


def _unregister(mapping: mmap.mmap, pointer: int) -> None:
    # Torch storage retains the numpy owner, including through cached UVA views.
    # Holding the mapping keeps it mapped until CUDA releases the registration.
    result = torch.cuda.cudart().cudaHostUnregister(pointer)
    if result.value != 0:
        logger.warning("Engram cudaHostUnregister failed: %s", result)


def register_host_memory(
    owner: np.ndarray, mapping: mmap.mmap, flags: int = 0
) -> torch.Tensor:
    """Pin a host mapping for UVA access and tie its lifetime to the tensor.

    Raises RuntimeError when CUDA does not pin it; the registration, if any,
    is released when the returned tensor's owner is dropped.
    """
    with warnings.catch_warnings():
        # A read-only mapping makes a non-writable array; writes then fault.
        warnings.filterwarnings("ignore", message="The given NumPy array")
        tensor = torch.from_numpy(owner)
    pointer = tensor.data_ptr()
    result = torch.cuda.cudart().cudaHostRegister(pointer, owner.nbytes, flags)
    if result.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {result}")
    finalizer = weakref.finalize(owner, _unregister, mapping, pointer)
    finalizer.atexit = False  # type: ignore[misc]
    # The UVA helper otherwise allocates a private pinned copy.
    if not tensor.is_pinned():
        raise RuntimeError("cudaHostRegister did not pin the Engram host mapping")
    return tensor


def read_only_registration_supported() -> bool:
    """Whether the current device can register a mapping without write access.

    Asked before registering, since a refused cudaHostRegister leaves its error
    pending, and torch then raises it at the next kernel launch.
    """
    value = ctypes.c_int()
    cudart = CudaRTLibrary().lib
    result = cudart.cudaDeviceGetAttribute(
        ctypes.byref(value),
        _CUDA_DEV_ATTR_HOST_REGISTER_READ_ONLY_SUPPORTED,
        torch.cuda.current_device(),
    )
    if result != 0:
        raise RuntimeError(f"cudaDeviceGetAttribute failed: {result}")
    return value.value != 0


def _mprotect(pointer: int, nbytes: int, prot: int) -> None:
    if _libc.mprotect(ctypes.c_void_p(pointer), ctypes.c_size_t(nbytes), prot) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"mprotect failed: {os.strerror(errno)}")


def register_read_only(
    owner: np.ndarray,
    mapping: mmap.mmap,
    supported: Callable[[], bool] = read_only_registration_supported,
    register: Callable[..., torch.Tensor] = register_host_memory,
) -> torch.Tensor:
    """Pin a read-only mapping, which stays read-only to the CPU.

    Where CUDA registers only writable memory (some virtualized GPUs, such as
    Verda's B300 VMs, report no read-only registration), the mapping is made
    writable for the registration and read-only again before anything else
    sees it. The pages stay pinned, and a CPU write still faults; only a write
    from the device is no longer refused.
    """
    if supported():
        return register(owner, mapping, _CUDA_HOST_REGISTER_READ_ONLY)
    logger.info_once(
        "The device cannot register read-only host memory; the Engram store "
        "registers its tables writable and maps them read-only again"
    )
    pointer = owner.ctypes.data
    _mprotect(pointer, owner.nbytes, mmap.PROT_READ | mmap.PROT_WRITE)
    try:
        return register(owner, mapping)
    finally:
        _mprotect(pointer, owner.nbytes, mmap.PROT_READ)


class PackedEngramStorage:
    """One table's FP8 rows and E8M0 scales packed in one host allocation."""

    def __init__(self, storage: torch.Tensor, num_rows: int, dim: int, block_size: int):
        weight_bytes = num_rows * dim
        self.weight = storage[:weight_bytes].view(torch.float8_e4m3fn).view(-1, dim)
        self.weight_scale_inv = storage[weight_bytes:].view(-1, dim // block_size)
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None

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


def _start_time(pid: int) -> str | None:
    """The kernel start time of ``pid``, which tells a live holder from a
    reused pid."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat.rsplit(")", 1)[1].split()[19]


def _pid_namespace() -> str:
    return os.readlink("/proc/self/ns/pid")


def _inode(fd: int) -> list[int]:
    stat = os.fstat(fd)
    return [stat.st_dev, stat.st_ino]


@contextmanager
def _locked_registry(
    lock_path: Path, registry_path: Path, blocking: bool = True
) -> Iterator[dict[str, Any] | None]:
    """The registry under the table's exclusive lock, rewritten if changed.

    Yields None when ``blocking`` is false and another process holds the lock.
    """
    with open(lock_path, "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield None
            return
        try:
            before = registry_path.read_text()
            registry = json.loads(before)
        except (OSError, ValueError):
            before, registry = None, {}
        yield registry
        after = json.dumps(registry, indent=1)
        if after != before:
            tmp = registry_path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(after)
            tmp.replace(registry_path)


def _release(fd: int, lock_path: Path, registry_path: Path) -> None:
    """Close this process's memfd and drop it from the holders.

    Best effort: a holder entry left behind (lock busy, or the process exits
    first) names an fd that is closed or reused, which ``_open`` skips.
    """
    try:
        with _locked_registry(lock_path, registry_path, blocking=False) as registry:
            if registry is not None:
                registry["holders"] = [
                    h
                    for h in registry.get("holders", [])
                    if (h["pid"], h["fd"]) != (os.getpid(), fd)
                ]
    except OSError as exc:
        logger.warning("Engram store %s deregistration failed: %s", registry_path, exc)
    os.close(fd)


class NodeSharedEngramStorage(PackedEngramStorage):
    """Host weights of one Engram table shared by every server on the host.

    Args:
        directory: The store directory, shared by the servers that share tables.
        identity: What makes two tables the same (model, revision, layer, shard
            bounds, shape); servers attach only to a table with equal identity.
        num_rows: Rows of this table shard.
        dim: Row width in FP8 bytes.
        block_size: Columns per E8M0 scale byte.
        register: Pins the read-only mapping; replaceable where CUDA is
            unavailable.

    """

    # A memfd mapping; shared_store requires use_thp=False, so its pages are small.
    huge_pages = False

    def __init__(
        self,
        directory: str,
        identity: dict[str, Any],
        num_rows: int,
        dim: int,
        block_size: int,
        register: Callable[[np.ndarray, mmap.mmap], torch.Tensor] = (
            register_read_only
        ),
    ) -> None:
        self._weight_bytes = num_rows * dim
        self.nbytes = self._weight_bytes + self._weight_bytes // block_size
        self.identity = dict(identity, nbytes=self.nbytes)
        # The file name is the identity, so a registry only describes its table.
        key = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode()
        ).hexdigest()[:24]
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._registry = self.directory / f"{key}.json"
        self._lock = self.directory / f"{key}.lock"
        self._fd, self.attached = self._open(key)
        self._finalizer = weakref.finalize(
            self, _release, self._fd, self._lock, self._registry
        )
        self._finalizer.atexit = False  # type: ignore[misc]
        # Read-only: only load_weight, through a transient writable mapping,
        # may change a table other servers can be serving.
        mapping = mmap.mmap(
            self._fd, self.nbytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ
        )
        super().__init__(
            register(np.frombuffer(mapping, dtype=np.uint8), mapping),
            num_rows,
            dim,
            block_size,
        )
        self.copied: list[str] = []

    def close(self) -> None:
        """Release this process's hold on the table; mappings stay valid."""
        self._finalizer()

    def _open(self, key: str) -> tuple[int, bool]:
        """Attach to a live holder's memfd, or create the table's memfd."""
        pid_namespace = _pid_namespace()
        with _locked_registry(self._lock, self._registry) as registry:
            assert registry is not None
            holders, unreachable, fd = [], [], None
            for holder in registry.get("holders", []):
                if holder.get("pid_ns") != pid_namespace:
                    # Its pid means nothing here, so it may be alive.
                    unreachable.append(f"pid {holder['pid']} in {holder['pid_ns']}")
                    continue
                if holder["pid"] == os.getpid() or (
                    _start_time(holder["pid"]) != holder["start"]
                ):
                    continue
                if fd is not None:
                    holders.append(holder)
                    continue
                try:
                    candidate = os.open(
                        f"/proc/{holder['pid']}/fd/{holder['fd']}",
                        os.O_RDWR | os.O_CLOEXEC,
                    )
                except FileNotFoundError:
                    continue  # It closed this fd.
                except OSError as exc:
                    unreachable.append(f"pid {holder['pid']}: {exc}")
                    continue
                if _inode(candidate) != registry.get("memfd"):
                    os.close(candidate)  # The fd number now names another file.
                    continue
                fd = candidate
                holders.append(holder)
            attached = fd is not None
            if not attached:
                if unreachable:
                    raise RuntimeError(
                        f"Engram store {self._registry} is held by processes this "
                        f"one cannot open ({'; '.join(unreachable)}). Servers "
                        "sharing a store must run as the same user in one PID "
                        "namespace; remove the file only if those holders are gone."
                    )
                fd = os.memfd_create(f"vllm_engram_{key}", os.MFD_CLOEXEC)
                os.ftruncate(fd, self.nbytes)
                holders = []
                registry.clear()
                registry.update(identity=self.identity, memfd=_inode(fd), loaded=[])
            # This process keeps the fd open so later servers can reach it.
            holders.append(
                {
                    "pid": os.getpid(),
                    "fd": fd,
                    "start": _start_time(os.getpid()),
                    "pid_ns": pid_namespace,
                }
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
        if param.data_ptr() == self.weight.data_ptr():
            name, offset = "weight", 0
        else:
            name, offset = "weight_scale_inv", self._weight_bytes
        shard = engram_head_shard(param, loaded_weight).contiguous()
        shard_bytes = shard.view(torch.uint8).reshape(-1)
        # Held through the copy, so a second server waits for the first.
        with _locked_registry(self._lock, self._registry) as registry:
            assert registry is not None
            if name in registry.get("loaded", []):
                table = param.data.view(torch.uint8).reshape(-1)
                if not torch.equal(table, shard_bytes):
                    raise RuntimeError(
                        f"Engram store {self._registry} was loaded from a "
                        f"different checkpoint ({name} differs)"
                    )
                logger.info(
                    "Engram store %s: %s already loaded by another server, not copied",
                    self._registry.name,
                    name,
                )
                return
            mapping = mmap.mmap(self._fd, self.nbytes, flags=mmap.MAP_SHARED)
            try:
                target = torch.frombuffer(
                    mapping, dtype=torch.uint8, count=shard_bytes.numel(), offset=offset
                )
                target.copy_(shard_bytes)
                del target
            finally:
                # Else it is unmapped once the last view is collected.
                with suppress(BufferError):
                    mapping.close()
            registry.setdefault("loaded", []).append(name)
            self.copied.append(name)
