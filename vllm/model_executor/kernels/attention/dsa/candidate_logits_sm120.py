# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-only indexer logits for DeepSeek V4.1 decode on SM120 (FP8 cache).

The two-level indexer's consumer layers keep only positions inside the
candidate blocks the candidate-source layer published. The dense path scores
the whole context with DeepGEMM's paged MQA logits, masks everything outside
the candidates to -inf and runs a top-k over ``max_model_len`` columns. This
module scores the candidate positions only, into a compact
``[rows, candidate_topk_blocks * candidate_block_size]`` matrix, and maps the
compact top-k back to request-local positions. The score of each candidate
position is the same function the dense kernel computes
(``sum_h w[h] * relu(q[h] . k) * k_scale``), so the selection matches the
dense path up to fp32 accumulation order.

The SM100 path (``indexer_sparse_logits``) needs DeepGEMM's MXFP4 sparse
kernels and DeepSelect; this one runs on the FP8 cache SM120 already uses.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _candidate_logits_kernel(
    q,  # [rows, H, D] fp8
    weights,  # [rows, H] fp32
    kv_cache,  # uint8, page p at p * stride_page: [P*D values][P fp32 scales]
    block_table,  # [B, max_pages] int32
    vis,  # [rows] int32: positions < vis[row] are visible
    candidates,  # [rows, KB] int32 block ids, -1 padded
    out,  # [rows, KB * CBS] fp32
    stride_q_row,
    stride_q_head,
    stride_w_row,
    stride_page,
    stride_bt_row,
    stride_vis,
    stride_c_row,
    stride_o_row,
    next_n,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    CBS: tl.constexpr,
    TILE_B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    brow = row // next_n
    n_vis = tl.load(vis + row * stride_vis)

    blk_off = tile * TILE_B + tl.arange(0, TILE_B)
    cand = tl.load(candidates + row * stride_c_row + blk_off)
    # [TILE_B * CBS] positions, block-major so output columns stay in order.
    cols = tl.arange(0, TILE_B * CBS)
    cand_rep = tl.reshape(
        tl.broadcast_to(cand[:, None], (TILE_B, CBS)), (TILE_B * CBS,)
    ).to(tl.int64)
    pos = cand_rep * CBS + (cols % CBS)
    valid = (cand_rep >= 0) & (pos < n_vis)
    out_ptr = out + row * stride_o_row + tile * (TILE_B * CBS) + cols
    if tl.max(valid.to(tl.int32), 0) == 0:
        tl.store(out_ptr, float("-inf"))
        return

    page = tl.load(block_table + brow * stride_bt_row + pos // PAGE, valid, other=0)
    slot = pos % PAGE
    base = page.to(tl.int64) * stride_page
    d = tl.arange(0, D)
    k = tl.load(
        kv_cache + base[:, None] + slot[:, None] * D + d[None, :],
        valid[:, None],
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    k_scale = tl.load(
        (kv_cache + base + PAGE * D + slot * 4).to(tl.pointer_type(tl.float32)),
        valid,
        other=0.0,
    )

    h = tl.arange(0, H)
    qv = tl.load(q + row * stride_q_row + h[:, None] * stride_q_head + d[None, :])
    w = tl.load(weights + row * stride_w_row + h)
    s = tl.dot(qv, tl.trans(k))  # [H, positions] fp32
    logits = tl.sum(tl.maximum(s, 0.0) * w[:, None], 0) * k_scale
    tl.store(out_ptr, tl.where(valid, logits, float("-inf")))


@triton.jit
def _map_topk_kernel(
    logits,
    indices,
    candidates,
    stride_l_row,
    stride_i_row,
    stride_c_row,
    K: tl.constexpr,
    CBS: tl.constexpr,
):
    # In place: compact column -> request-local position, -1 for columns
    # that were never visible (top-k of a row with < K visible candidates).
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, K)
    idx = tl.load(indices + row * stride_i_row + j)
    ok = idx >= 0
    v = tl.load(logits + row * stride_l_row + idx, ok, other=float("-inf"))
    blk = tl.load(candidates + row * stride_c_row + idx // CBS, ok, other=-1)
    pos = blk * CBS + idx % CBS
    tl.store(indices + row * stride_i_row + j, tl.where(v == float("-inf"), -1, pos))


def candidate_logits(
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    vis: torch.Tensor,
    candidates: torch.Tensor,
    candidate_block_size: int,
    next_n: int,
    out: torch.Tensor | None = None,
    tile_blocks: int = 16,
) -> torch.Tensor:
    """Compact logits ``[rows, KB * CBS]`` over each row's candidate blocks.

    q: [rows, H, D] float8_e4m3fn; weights: [rows, H] fp32 (Q scale folded);
    kv_cache: uint8 with ``kv_cache.shape[1]`` tokens per page laid out as
    the SM120 FP8 indexer cache (values, then one fp32 scale per token);
    block_table: [B, max_pages] with row r reading request r // next_n;
    vis: [rows] int32 visible lengths; candidates: [rows, KB] int32.
    """
    rows, H, D = q.shape
    KB = candidates.shape[1]
    page = kv_cache.shape[1]
    assert KB % tile_blocks == 0
    assert q.stride(-1) == 1 and weights.stride(-1) == 1
    assert block_table.stride(-1) == 1 and candidates.stride(-1) == 1
    if out is None:
        out = torch.empty(
            (rows, KB * candidate_block_size), dtype=torch.float32, device=q.device
        )
    if rows == 0:
        return out
    _candidate_logits_kernel[(rows, KB // tile_blocks)](
        q,
        weights,
        kv_cache,
        block_table,
        vis,
        candidates,
        out,
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        kv_cache.stride(0),
        block_table.stride(0),
        vis.stride(0),
        candidates.stride(0),
        out.stride(0),
        next_n,
        H=H,
        D=D,
        PAGE=page,
        CBS=candidate_block_size,
        TILE_B=tile_blocks,
        num_warps=4,
    )
    return out


def candidate_topk(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    candidate_block_size: int,
    topk: int,
    out: torch.Tensor,
    next_n: int,
    topk_op=None,
) -> None:
    """Top-``topk`` request-local positions from compact candidate logits
    into ``out`` ([rows, topk] int32); -1 where a row has fewer visible
    candidates than ``topk``. ``topk_op`` is the indexer's decode top-k
    (`SparseIndexerTopk`); torch.topk when None."""
    rows, width = logits.shape
    if rows == 0:
        return
    if topk_op is None:
        out.copy_(logits.topk(topk, dim=-1, sorted=False).indices)
    else:
        lens = torch.full(
            (rows // next_n, next_n), width, dtype=torch.int32, device=logits.device
        )
        topk_op(logits, lens, next_n, out, topk, width)
    _map_topk_kernel[(rows,)](
        logits,
        out,
        candidates,
        logits.stride(0),
        out.stride(0),
        candidates.stride(0),
        K=topk,
        CBS=candidate_block_size,
    )
