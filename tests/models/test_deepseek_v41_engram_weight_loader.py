# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loading an engram table's shard of a safetensors checkpoint."""

import os
import sys

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)
from vllm.models.deepseek_v41.common import engram

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="direct I/O and mincore are Linux-only"
)

# Above the size the loader reads directly, and more than one aligned buffer.
ROWS, DIM = 32768, 4096
SKIPPED_ROWS = 96


@pytest.fixture
def checkpoint(tmp_path):
    """A shard holding a table after a tensor of odd size, so that the table
    starts off a block boundary, with none of it left in the page cache."""
    expected = torch.randint(0, 256, (ROWS, DIM), dtype=torch.uint8)
    path = tmp_path / "model-00001-of-00001.safetensors"
    save_file({"a": torch.zeros(1237, dtype=torch.uint8), "table": expected}, path)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return str(path), expected


def _mapped_table(path: str) -> torch.Tensor:
    """The table as the model loader hands it to a weight loader."""
    return dict(safetensors_weights_iterator([path], use_tqdm_on_load=False))["table"]


def _param() -> torch.nn.Parameter:
    param = torch.nn.Parameter(
        torch.zeros(ROWS - SKIPPED_ROWS, DIM, dtype=torch.uint8), requires_grad=False
    )
    param.engram_vocab_start = SKIPPED_ROWS
    return param


def _resident(table: torch.Tensor) -> float:
    return engram._resident_share(table.data_ptr(), table.numel())


def test_a_table_on_disk_is_loaded_without_filling_the_page_cache(checkpoint):
    path, expected = checkpoint
    table = _mapped_table(path)
    # Opening the shard reads ahead a few MiB past its header, no more.
    on_disk = _resident(table) < 0.5
    param = _param()

    engram._engram_head_shard_weight_loader(param, table)

    assert torch.equal(param.data, expected[SKIPPED_ROWS:])
    # A filesystem that keeps its files in memory (tmpfs) has nothing to show.
    if on_disk:
        assert _resident(table) < 0.5


def test_a_table_in_memory_is_copied_from_it(checkpoint):
    path, expected = checkpoint
    table = _mapped_table(path)
    assert torch.equal(table, expected)  # reads it into the page cache
    assert _resident(table) == 1.0
    param = _param()

    engram._engram_head_shard_weight_loader(param, table)

    assert torch.equal(param.data, expected[SKIPPED_ROWS:])
    assert _resident(table) == 1.0


def test_the_backing_file_of_a_mapping(checkpoint):
    path, _ = checkpoint
    table = _mapped_table(path)

    found = engram._backing_file(table.data_ptr())

    assert found is not None and found[0] == path
    with open(path, "rb") as file:
        file.seek(found[1])
        assert file.read(DIM) == table[0].numpy().tobytes()
    assert engram._backing_file(torch.zeros(8).data_ptr()) is None
