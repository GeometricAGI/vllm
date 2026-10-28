# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Converting MoE weights to Humming's format a few experts at a time must give
the whole-layer result, so that a large layer converts within the GPU memory
left free while the model loads."""

import dataclasses

import torch

from vllm.model_executor.layers.quantization.utils.humming import moe


@dataclasses.dataclass(frozen=True)
class _Schema:
    """Stands in for a Humming weight schema: offsets every code by one, and
    reports `kind` as the converted schema (per chunk if `kinds` is set)."""

    kind: str = "converted"
    kinds: tuple[str, ...] | None = None

    def convert_humming(self, tensors, num_experts, **kwargs):
        out = {k: v + 1 for k, v in tensors.items()}
        kind = self.kind
        if self.kinds is not None:
            kind = self.kinds[int(tensors["weight"][0, 0].item()) // 32]
        return _Schema(kind), out


def _tensors(num_experts):
    weight = torch.arange(num_experts, dtype=torch.int32)[:, None].expand(
        num_experts, 3
    )
    return {"weight": weight.contiguous(), "scale": torch.rand(num_experts, 2)}


def test_chunked_conversion_matches_whole_layer():
    tensors = _tensors(100)
    whole_schema, whole = _Schema().convert_humming(tensors, num_experts=100)
    schema, chunked = moe._convert_weight_in_expert_chunks(_Schema(), tensors, 100)
    assert schema == whole_schema
    assert chunked.keys() == whole.keys()
    for k in whole:
        torch.testing.assert_close(chunked[k], whole[k], rtol=0, atol=0)


def test_chunks_that_disagree_on_the_schema_fall_back_to_one_call():
    tensors = _tensors(100)
    schema = _Schema(kinds=("a", "b", "a", "b"))
    converted_schema, out = moe._convert_weight_in_expert_chunks(schema, tensors, 100)
    assert converted_schema.kind == "a"  # the single whole-layer call: chunk 0's view
    torch.testing.assert_close(out["weight"], tensors["weight"] + 1, rtol=0, atol=0)


def test_chunks_that_disagree_on_a_shape_fall_back_to_one_call():
    """Humming may pick an output's shape from the data (one scale per expert
    when its stacked scales are equal), so chunks can disagree on it."""
    tensors = _tensors(100)

    @dataclasses.dataclass(frozen=True)
    class DataDependentShape(_Schema):
        def convert_humming(self, tensors, num_experts, **kwargs):
            schema, out = super().convert_humming(tensors, num_experts, **kwargs)
            if int(tensors["weight"][0, 0].item()) >= 32:
                out["scale"] = out["scale"][:, :1]
            return schema, out

    _, out = moe._convert_weight_in_expert_chunks(DataDependentShape(), tensors, 100)
    torch.testing.assert_close(out["scale"], tensors["scale"] + 1, rtol=0, atol=0)


def test_tensors_not_per_expert_convert_in_one_call():
    tensors = _tensors(100)
    tensors["global_scale"] = torch.ones(1)
    calls = []

    @dataclasses.dataclass(frozen=True)
    class Counting(_Schema):
        def convert_humming(self, tensors, num_experts, **kwargs):
            calls.append(num_experts)
            return super().convert_humming(tensors, num_experts, **kwargs)

    moe._convert_weight_in_expert_chunks(Counting(), tensors, 100)
    assert calls == [100]
