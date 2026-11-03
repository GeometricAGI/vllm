# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recoding symmetric integer MoE weights a few experts at a time must give the
whole-layer result, so that a large layer converts within the GPU memory left
free while the model loads."""

import dataclasses

import torch

from vllm.model_executor.layers.quantization.utils.humming import moe


@dataclasses.dataclass(frozen=True)
class _Schema:
    """Stands in for ``HummingWeightSchema.convert_symmetric_integer_weight``:
    keeps the codes (returning itself and its input) if any code is 0, else
    recodes ``weight`` and passes the other tensors through."""

    kind: str = "offset"

    def convert_symmetric_integer_weight(self, tensors):
        if torch.any(tensors["weight"] == 0):
            return self, tensors
        out = tensors.copy()
        out["weight"] = tensors["weight"] * 2
        return dataclasses.replace(self, kind="signed"), out


def _tensors(num_experts):
    weight = torch.arange(1, num_experts + 1, dtype=torch.int32)[:, None]
    return {
        "weight": weight.expand(num_experts, 3).contiguous(),
        "weight_scale": torch.rand(num_experts, 2),
    }


def test_chunked_recoding_matches_whole_layer_and_keeps_passthrough_tensors():
    tensors = _tensors(100)
    whole_schema, whole = _Schema().convert_symmetric_integer_weight(tensors)

    schema, chunked = moe._convert_symmetric_integer_weight_in_expert_chunks(
        _Schema(), tensors, 100
    )

    assert schema == whole_schema
    assert chunked.keys() == whole.keys()
    torch.testing.assert_close(chunked["weight"], whole["weight"], rtol=0, atol=0)
    assert chunked["weight_scale"] is tensors["weight_scale"]


def test_a_zero_code_in_any_chunk_keeps_the_whole_layers_codes():
    tensors = _tensors(100)
    tensors["weight"][70, 1] = 0
    schema = _Schema()

    converted_schema, out = moe._convert_symmetric_integer_weight_in_expert_chunks(
        schema, tensors, 100
    )

    assert converted_schema is schema
    assert out is tensors


def test_a_chunk_output_not_per_expert_falls_back_to_one_call():
    """A chunk whose output collapses the expert dim must not be broadcast
    over its experts."""
    calls = []

    @dataclasses.dataclass(frozen=True)
    class CollapsesLaterChunks(_Schema):
        def convert_symmetric_integer_weight(self, tensors):
            calls.append(tensors["weight"].size(0))
            schema, out = super().convert_symmetric_integer_weight(tensors)
            if tensors["weight"].size(0) > 32 or int(tensors["weight"][0, 0]) == 1:
                return schema, out
            out["weight"] = out["weight"][:1]
            return schema, out

    tensors = _tensors(100)
    _, out = moe._convert_symmetric_integer_weight_in_expert_chunks(
        CollapsesLaterChunks(), tensors, 100
    )

    assert calls == [32, 32, 100]
    torch.testing.assert_close(out["weight"], tensors["weight"] * 2, rtol=0, atol=0)
