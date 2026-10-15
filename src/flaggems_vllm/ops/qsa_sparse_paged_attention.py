# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors

"""QSA sparse paged attention ported from vLLM f47792af7f.

Launch geometry: the host fixes the split-K count from an H800-tuned table
(H20-era and other-arch tables are kept for non-H800 devices), while
BLOCK_N / num_warps / num_stages go through the FlagGems tuned-config
pipeline (runtime.get_tuned_config + libtuner) with a key on the real
performance dimensions (rows, heads, GQA ratio, head dim, selection width,
page size, split count and key cache strides). KV dtype is part of the key
automatically via tensor dtypes.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


@lru_cache(maxsize=1)
def _is_sm120() -> bool:
    """True on sm_120 (RTX PRO 6000 Blackwell): selects the sm_120 tuning table."""
    return (
        runtime.device.vendor_name == "nvidia"
        and runtime.torch_device_fn.get_device_capability() == (12, 0)
    )


@lru_cache(maxsize=1)
def _is_sm90() -> bool:
    """True on sm_90 (H100/H200/H20): selects the sm_90 tuning table."""
    return (
        runtime.device.vendor_name == "nvidia"
        and runtime.torch_device_fn.get_device_capability() == (9, 0)
    )


@lru_cache(maxsize=1)
def _is_h800() -> bool:
    """True on H800 (sm_90): the split table below was retuned on H800."""
    if not _is_sm90():
        return False
    try:
        name = runtime.torch_device_fn.get_device_name()
    except Exception:  # noqa: BLE001 - fall back to the generic sm_90 table
        return False
    return "H800" in name


@triton.jit(do_not_specialize=["num_rows", "num_requests", "num_cache_blocks"])
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    softmax_scale,
    output_scale,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FP8: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)

    # The packed selection buffer carries one TRAILING COUNT COLUMN per row
    # (column TOPK of a TOPK+1-wide buffer): the row's valid-entry count,
    # written by the expand kernel. It is never a token index — the tile loop
    # and the index load below only ever cover columns [0, TOPK).
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    tile_end = tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N)
    # Splits without work exit before touching Q; split 0 always stays so
    # rows with zero or one tile still get their (zero or gated) final output.
    if split_id > 0 and split_id >= tile_end:
        return

    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    # softmax_scale is the host-side attention scale (1/sqrt(head_dim), with the
    # fp8 K dequant scale already pre-multiplied in); convert to log2 units once
    # here for the exp2-based online softmax.
    score_scale = softmax_scale * 1.4426950408889634

    for tile in range(split_id, tile_end, NUM_SPLITS):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        if IS_FP8:
            # e4m3 -> Q dtype is exact; keep the QK dot in Q's dtype (fp8 QK
            # measured slower here and less accurate).
            keys = keys.to(query.dtype)
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16; for fp8
        # caches the K dequant scale is already folded into softmax_scale on the
        # host.
        scores *= score_scale
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        if IS_FP8:
            # Dequant V to fp16 (not bf16) for the PV dot: P <= 1 (online
            # softmax) so fp16 has the range, its wider mantissa is more
            # accurate, and the fp8->fp16 upcast with an fp16 PV dot is faster.
            values = values.to(tl.float16)
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    # Fold the fp8 V dequant scale (output_scale, 1.0 for bf16) into a per-row
    # reciprocal normalizer, so the output is a per-row multiply rather than a
    # HEAD_DIM-wide scale. The split-K LSE below keeps the unscaled normalizer,
    # so the merge stays correct.
    inv_normalizer = output_scale / tl.maximum(normalizer, 1.0e-20)
    normalized_output = tl.where(
        has_values[:, None],
        accumulator * inv_normalizer[:, None],
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    # Rows with at most one tile (including empty selections) are finalized
    # directly by split 0 — BF16 rounding, then the gate in FP32 — and the
    # merge kernel skips them.
    if NUM_SPLITS == 1 or tile_end <= 1:
        # Preserve the unfused path's BF16 attention-output rounding before
        # applying the gate in FP32.
        normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
        output_gate = tl.load(
            output_gate_ptr
            + row * stride_output_gate_row
            + (first_head + head_offsets[:, None]) * stride_output_gate_head
            + dim_offsets[None, :],
            mask=output_mask,
            other=0.0,
        ).to(tl.float32)
        normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        tl.store(
            partial_output_ptr
            + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    indices_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    stride_indices_row,
    num_rows,
    TOPK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    # Rows with at most one valid tile were finalized directly by split 0 of
    # the main kernel (including empty selections, which become zeros); there
    # is nothing to merge for them. The same count also tells how many splits
    # actually wrote partials, so reads never touch unwritten (stale)
    # workspace — CUDA-graph replays with different contents stay correct.
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    num_valid_tiles = tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N)
    if num_valid_tiles <= 1:
        return
    merge_splits = tl.minimum(num_valid_tiles, NUM_SPLITS)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < merge_splits
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    split_rows = split_offsets.to(tl.int64) * num_rows + row
    partial_output = tl.load(
        partial_output_ptr
        + (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    # Preserve the unfused path's BF16 attention-output rounding before
    # applying the gate in FP32.
    merged = merged.to(output_ptr.dtype.element_ty)
    output_gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head * stride_output_gate_head
        + dim_offsets
    ).to(tl.float32)
    merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


# Tuned-config entries. The yaml search space is the union of the H800 scan
# winners and the previous H20/sm_120 table values, so machines other than
# H800 can still bench-pick their own best; vendors without a yaml entry fall
# back to the default config below.
_QSA_MAIN_TUNED = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("qsa_sparse_paged_attention")
        or [triton.Config(kwargs={"BLOCK_N": 64}, num_warps=2, num_stages=2)],
        key=[
            "num_rows",
            "NUM_QUERY_HEADS",
            "GROUP_SIZE",
            "HEAD_DIM",
            "TOPK",
            "PAGE_SIZE",
            "NUM_SPLITS",
            "stride_k_head",
            "stride_k_token",
            "stride_k_block",
            "stride_indices_row",
        ],
        strategy=[
            "log",
            "default",
            "default",
            "default",
            "default",
            "default",
            "default",
            "default",
            "default",
            "default",
            "default",
        ],
        warmup=10,
        rep=30,
    )(_qsa_sparse_paged_gqa_splitk_kernel)
)

_QSA_MERGE_TUNED = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("qsa_sparse_paged_attention_merge")
        or [triton.Config(kwargs={}, num_warps=2, num_stages=1)],
        key=["NUM_SPLITS", "HEAD_DIM", "TOPK", "BLOCK_N"],
        strategy=["log", "default", "default", "default"],
        warmup=10,
        rep=30,
    )(_qsa_merge_splitk_kernel)
)


def _select_sm120_splits(
    base_programs: int, use_prefill_config: bool, is_fp8: bool
) -> int:
    """Split count from the sm_120 (RTX PRO 6000 Blackwell) tuning table."""
    if base_programs > 2048:
        return 1
    if is_fp8:
        if base_programs <= 24:
            return 64
        if base_programs <= 32:
            return 16
        if base_programs <= 64:
            return 8
        if base_programs <= 256:
            return 8
        return 4
    if base_programs <= 24:
        return 64
    if base_programs <= 128:
        return 8
    if base_programs <= 256:
        return 8
    return 4


def _select_sm90_splits(
    base_programs: int, use_prefill_config: bool, is_fp8: bool
) -> int:
    """Split count from the legacy sm_90 (H20) tuning table."""
    if base_programs > 2048:
        return 1
    if is_fp8:
        if base_programs <= 24:
            return 64
        if base_programs <= 64:
            return 16
        if base_programs <= 128:
            return 8
        if base_programs <= 256:
            return 8
        return 4
    if base_programs <= 32:
        return 64
    if base_programs <= 64:
        return 16
    if base_programs <= 256:
        return 8
    return 4


def _select_h800_splits(
    base_programs: int, use_prefill_config: bool, is_fp8: bool
) -> int:
    """Split count retuned on H800 (sm_90) for BF16 decode/prefill.

    From the H800 config scan (selection width 2051, head dim 256): decode
    sized grids (<=32 programs) want the selection fully split; the 256-512
    program region wants 2-4 splits; larger grids run single-split. FP8 is
    not the optimization target and keeps the legacy table.
    """
    if is_fp8:
        return _select_sm90_splits(base_programs, use_prefill_config, is_fp8)
    if base_programs > 512:
        return 1
    if base_programs > 256:
        return 4
    if base_programs > 32:
        return 2
    return 64


def _select_splits(
    num_rows: int,
    num_kv_heads: int,
    use_prefill_config: bool,
    is_fp8: bool,
) -> int:
    """Target split count for the main kernel (clamped later by tile count)."""
    base_programs = num_rows * num_kv_heads
    if _is_h800():
        return _select_h800_splits(base_programs, use_prefill_config, is_fp8)
    if _is_sm120():
        return _select_sm120_splits(base_programs, use_prefill_config, is_fp8)
    if _is_sm90():
        return _select_sm90_splits(base_programs, use_prefill_config, is_fp8)
    return _select_sm120_splits(base_programs, use_prefill_config, is_fp8)


class _QsaPlan:
    """Reusable launch state for one input signature.

    Resolved once through the libtuner-wrapped entries (which performs the
    tuned-config selection and returns the compiled kernel); subsequent calls
    with the same metadata replay the compiled kernels directly, skipping the
    per-argument dispatcher overhead that dominates small-batch latency.
    """

    __slots__ = (
        "main_kernel",
        "main_grid",
        "main_template",
        "main_slots",
        "merge_kernel",
        "merge_grid",
        "merge_template",
        "merge_slots",
        "num_splits",
        "partial_shape",
        "lse_shape",
        "softmax_scale",
        "output_scale",
    )


# Bounded dict of _QsaPlan keyed by full launch metadata. Validation runs in
# full on the first call for each signature; the key contains every field the
# validation depends on, so cache hits imply identical validation outcomes.
_PLANS: dict = {}
_MAX_PLANS = 512


def _plan_key(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    output_gate: torch.Tensor,
    out: torch.Tensor | None,
    use_prefill_config: bool,
    k_scale: float | None,
    v_scale: float | None,
) -> tuple:
    align = 15  # triton specializes pointer divisibility by 16
    return (
        q.device,
        q.shape,
        q.dtype,
        q.stride(),
        q.data_ptr() & align,
        k_cache.shape,
        k_cache.dtype,
        k_cache.stride(),
        k_cache.data_ptr() & align,
        v_cache.dtype,
        v_cache.stride(),
        v_cache.data_ptr() & align,
        logical_indices.shape,
        logical_indices.stride(),
        logical_indices.data_ptr() & align,
        block_table.shape,
        block_table.stride(),
        block_table.data_ptr() & align,
        token_to_req.shape,
        token_to_req.stride(),
        token_to_req.data_ptr() & align,
        output_gate.shape,
        output_gate.stride(),
        output_gate.data_ptr() & align,
        use_prefill_config,
        k_scale,
        v_scale,
        (
            None
            if out is None
            else (out.shape, out.dtype, out.stride(), out.data_ptr() & align)
        ),
    )


def _build_plan(
    key: tuple,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    output_gate: torch.Tensor,
    out: torch.Tensor | None,
    use_prefill_config: bool,
    k_scale: float | None,
    v_scale: float | None,
) -> _QsaPlan:
    """Validate the inputs and resolve the tuned kernels for this signature."""
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] < 2:
        raise ValueError(
            "QSA packed indices need selection columns plus the count column"
        )
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == v_cache.dtype
    is_fp8 = k_cache.dtype == torch.float8_e4m3fn
    if is_fp8:
        assert k_scale is not None and v_scale is not None
        # Host pre-multiply: fold the K dequant scale into the attention scale
        # and pass V's dequant scale as the kernel's output scale.
        softmax_scale = (head_dim**-0.5) * float(k_scale)
        output_scale = float(v_scale)
    else:
        assert k_cache.dtype == torch.bfloat16
        softmax_scale = head_dim**-0.5
        output_scale = 1.0
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.device == logical_indices.device == block_table.device
    assert q.device == token_to_req.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1

    if out is not None:
        if out.shape != q.shape:
            raise ValueError("QSA sparse output must match its query")
        assert out.dtype == q.dtype and out.device == q.device
        assert out.stride(2) == 1
    assert output_gate.is_contiguous()
    if not q.shape[0]:
        return None  # empty batch: caller returns immediately

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    selection_width = logical_indices.shape[1] - 1  # trailing column is the count
    target_splits = _select_splits(
        q.shape[0], k_cache.shape[2], use_prefill_config, is_fp8
    )
    # Never more splits than the smallest candidate BLOCK_N can feed; splits
    # beyond the effective tile count of the tuned BLOCK_N exit empty in the
    # kernel (they still write mergeable partials, so results stay correct).
    num_splits = min(target_splits, triton.cdiv(selection_width, 32))

    num_rows = q.shape[0]
    num_kv_heads = k_cache.shape[2]
    out_arg = out if out is not None else torch.empty_like(q)
    gate_arg = output_gate  # contiguous; strides below match view_as(q)

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out_arg
        partial_lse = out_arg
        partial_shape = None
        lse_shape = None
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_shape = (num_splits, num_rows, q.shape[1], head_dim)
        lse_shape = (num_splits, num_rows, q.shape[1])
        partial_output = torch.empty(
            partial_shape, dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)

    main_kwargs = dict(
        q_ptr=q,
        k_cache_ptr=k_cache,
        v_cache_ptr=v_cache,
        indices_ptr=logical_indices,
        block_table_ptr=block_table,
        token_to_req_ptr=token_to_req,
        partial_output_ptr=partial_output,
        partial_lse_ptr=partial_lse,
        output_ptr=out_arg,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        output_gate_ptr=gate_arg,
        stride_q_row=q.stride(0),
        stride_q_head=q.stride(1),
        stride_k_block=k_cache.stride(0),
        stride_k_token=k_cache.stride(1),
        stride_k_head=k_cache.stride(2),
        stride_v_block=v_cache.stride(0),
        stride_v_token=v_cache.stride(1),
        stride_v_head=v_cache.stride(2),
        stride_indices_row=logical_indices.stride(0),
        stride_table_req=block_table.stride(0),
        stride_output_row=out_arg.stride(0),
        stride_output_head=out_arg.stride(1),
        stride_output_gate_row=gate_arg.stride(0),
        stride_output_gate_head=gate_arg.stride(1),
        num_rows=num_rows,
        num_cache_blocks=k_cache.shape[0],
        num_requests=block_table.shape[0],
        TOPK=selection_width,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_M=block_m,
        IS_FP8=is_fp8,
    )
    main_grid = (num_rows, num_kv_heads, num_splits, 1, 1)
    main_kernel, main_constexprs = _QSA_MAIN_TUNED[main_grid](**main_kwargs)
    # NOTE: for a contiguous gate, view_as(q) reproduces gate's own strides.
    main_names = list(_QSA_MAIN_TUNED.signature.parameters.keys())
    main_template = [
        main_kwargs[n] if n in main_kwargs else main_constexprs[n] for n in main_names
    ]
    main_slots = tuple(
        main_names.index(n)
        for n in (
            "q_ptr",
            "k_cache_ptr",
            "v_cache_ptr",
            "indices_ptr",
            "block_table_ptr",
            "token_to_req_ptr",
            "partial_output_ptr",
            "partial_lse_ptr",
            "output_ptr",
            "output_gate_ptr",
        )
    )

    plan = _QsaPlan()
    plan.main_kernel = main_kernel
    plan.main_grid = main_grid
    plan.main_template = main_template
    plan.main_slots = main_slots
    plan.merge_kernel = None
    plan.merge_grid = None
    plan.merge_template = None
    plan.merge_slots = None
    plan.num_splits = num_splits
    plan.partial_shape = partial_shape
    plan.lse_shape = lse_shape
    plan.softmax_scale = softmax_scale
    plan.output_scale = output_scale

    if num_splits > 1:
        merge_kwargs = dict(
            partial_output_ptr=partial_output,
            partial_lse_ptr=partial_lse,
            output_ptr=out_arg,
            output_gate_ptr=gate_arg,
            indices_ptr=logical_indices,
            stride_output_row=out_arg.stride(0),
            stride_output_head=out_arg.stride(1),
            stride_output_gate_row=gate_arg.stride(0),
            stride_output_gate_head=gate_arg.stride(1),
            stride_indices_row=logical_indices.stride(0),
            num_rows=num_rows,
            TOPK=selection_width,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=num_splits,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            BLOCK_N=main_constexprs["BLOCK_N"],
        )
        merge_grid = (num_rows, q.shape[1], 1, 1, 1)
        merge_kernel, merge_constexprs = _QSA_MERGE_TUNED[merge_grid](**merge_kwargs)
        merge_names = list(_QSA_MERGE_TUNED.signature.parameters.keys())
        plan.merge_kernel = merge_kernel
        plan.merge_grid = merge_grid
        plan.merge_template = [
            merge_kwargs[n] if n in merge_kwargs else merge_constexprs[n]
            for n in merge_names
        ]
        plan.merge_slots = tuple(
            merge_names.index(n)
            for n in (
                "partial_output_ptr",
                "partial_lse_ptr",
                "output_ptr",
                "output_gate_ptr",
                "indices_ptr",
            )
        )
    return plan


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    *,
    output_gate: torch.Tensor,
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 or FP8-e4m3 K/V caches.

    With fp8 caches, k_scale/v_scale are the layer's per-tensor dequant scales
    as host floats (e.g. layer._k_scale_float/_v_scale_float): k_scale is
    pre-multiplied into the softmax scale and v_scale becomes the kernel's
    output scale, so the kernel needs no device scale buffers.

    logical_indices is the PACKED selection buffer: [rows, selection_width + 1]
    with the trailing column holding each row's valid-entry count (written by
    the expand kernel; never a token index). The kernel reads it as the
    tile-loop bound. use_prefill_config only steers the split-count table; see
    _select_splits.
    """
    key = _plan_key(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        output_gate,
        out,
        use_prefill_config,
        k_scale,
        v_scale,
    )
    plan = _PLANS.get(key)
    if plan is None:
        if len(_PLANS) >= _MAX_PLANS:
            _PLANS.clear()
        plan = _build_plan(
            key,
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            output_gate,
            out,
            use_prefill_config,
            k_scale,
            v_scale,
        )
        if plan is None:
            # Empty batch: nothing to compute, but the output must still be a
            # valid tensor of the right shape.
            return out if out is not None else torch.empty_like(q)
        _PLANS[key] = plan

    if out is None:
        out = torch.empty_like(q)
    if plan.num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        partial_output = torch.empty(
            plan.partial_shape, dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(plan.lse_shape, dtype=torch.float32, device=q.device)

    main_args = plan.main_template.copy()
    for slot, tensor in zip(
        plan.main_slots,
        (
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            partial_output,
            partial_lse,
            out,
            output_gate,
        ),
    ):
        main_args[slot] = tensor
    plan.main_kernel[plan.main_grid](*main_args)

    if plan.num_splits == 1:
        return out

    merge_args = plan.merge_template.copy()
    for slot, tensor in zip(
        plan.merge_slots,
        (partial_output, partial_lse, out, output_gate, logical_indices),
    ):
        merge_args[slot] = tensor
    plan.merge_kernel[plan.merge_grid](*merge_args)
    return out
