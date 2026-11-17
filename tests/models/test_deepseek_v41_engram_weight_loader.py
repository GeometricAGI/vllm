# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loading an engram table's shard of the checkpoint."""

import os

import pytest
import torch

from vllm.models.deepseek_v41.common import engram

ROWS, DIM = 4096, 4096


def _param(rows: int, dim: int, start: int, dtype: torch.dtype) -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.zeros(rows, dim, dtype=dtype), requires_grad=False)
    param.engram_vocab_start = start
    return param


@pytest.fixture
def table(tmp_path):
    """A table in a file behind a header, as in a shard, and what it holds."""
    expected = torch.randint(0, 255, (ROWS, DIM), dtype=torch.uint8)
    path = tmp_path / "table.bin"
    header = b"h" * 1234
    path.write_bytes(header + expected.numpy().tobytes())
    mapped = torch.from_file(
        str(path), shared=False, size=len(header) + ROWS * DIM, dtype=torch.uint8
    )[len(header) :].view(ROWS, DIM)
    return path, mapped, expected


def test_a_table_in_memory_is_copied(table, monkeypatch):
    _, mapped, expected = table
    monkeypatch.setattr(engram, "_read_from_file", pytest.fail)
    param = _param(rows=ROWS - 96, dim=DIM, start=96, dtype=torch.uint8)

    engram._engram_head_shard_weight_loader(param, mapped)

    assert torch.equal(param.data, expected[96:])


def test_a_table_not_in_memory_is_read_from_its_file(table, monkeypatch):
    path, mapped, expected = table
    monkeypatch.setattr(engram, "_ENGRAM_READ_BYTES", 1 << 20)
    monkeypatch.setattr(engram, "_resident_share", lambda pointer, nbytes: 0.0)
    read = []
    direct = engram._read_from_file

    def reading(destination, file, offset):
        # The test's filesystem may want aligned direct reads; read it buffered.
        monkeypatch.setattr(os, "O_DIRECT", 0)
        read.append((file, offset))
        return direct(destination, file, offset)

    monkeypatch.setattr(engram, "_read_from_file", reading)
    param = _param(rows=ROWS - 96, dim=DIM, start=96, dtype=torch.uint8)

    engram._engram_head_shard_weight_loader(param, mapped)

    assert read == [(str(path), 1234 + 96 * DIM)]
    assert torch.equal(param.data, expected[96:])


def test_a_filesystem_that_cannot_read_directly_is_copied(table, monkeypatch):
    _, mapped, expected = table
    monkeypatch.setattr(engram, "_ENGRAM_READ_BYTES", 1 << 20)
    monkeypatch.setattr(engram, "_resident_share", lambda pointer, nbytes: 0.0)

    def failing(destination, file, offset):
        destination.fill_(7)  # what a read that failed part-way leaves
        return False

    monkeypatch.setattr(engram, "_read_from_file", failing)
    param = _param(rows=ROWS, dim=DIM, start=0, dtype=torch.uint8)

    engram._engram_head_shard_weight_loader(param, mapped)

    assert torch.equal(param.data, expected)


def test_the_backing_file_of_a_mapping(table):
    path, mapped, _ = table

    assert engram._backing_file(mapped.data_ptr()) == (str(path), 1234)
    assert engram._backing_file(torch.zeros(8).data_ptr()) is None


def test_resident_share_does_not_touch_the_mapping(table):
    _, mapped, expected = table
    nbytes = ROWS * DIM

    share = engram._resident_share(mapped.data_ptr(), nbytes)

    assert 0.0 <= share <= 1.0
    assert torch.equal(mapped, expected)
    assert engram._resident_share(mapped.data_ptr(), nbytes) == 1.0
