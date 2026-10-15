# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors

"""QSA attention comparison, using vLLM v0.31.0/main as the baseline.

Adapted from vLLM tests/models/qwen4_exp/test_qsa_reference.py at f47792af7f.
Inputs use synthetic token indices directly, without indexer or block expansion.
The baseline already applies the output gate.
"""

import pytest
import torch

import flaggems_vllm

from . import conftest as cfg


def _qsa_sparse_paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    softmax_scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """Dense reference for QSA sparse paged attention.

    Mirrors the kernel's dequant: fp8-e4m3 K/V caches are dequantized with the
    per-tensor k_scale/v_scale host floats; bf16 caches use unit scales.
    """
    output = torch.zeros_like(q)
    repeats = q.shape[1] // k_cache.shape[2]
    page_size = k_cache.shape[1]
    for row in range(q.shape[0]):
        logical = logical_indices[row]
        logical = logical[logical >= 0].long()
        if not logical.numel():
            continue
        request = token_to_req[row].long()
        pages = block_table[request, logical // page_size].long()
        offsets = logical % page_size
        keys = (k_cache[pages, offsets].float() * k_scale).repeat_interleave(
            repeats, dim=1
        )
        values = (v_cache[pages, offsets].float() * v_scale).repeat_interleave(
            repeats, dim=1
        )
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys)
        probabilities = torch.softmax(scores * softmax_scale, dim=-1)
        output[row] = torch.einsum("hk,khd->hd", probabilities, values).to(q.dtype)
    return output


def _make_logical_indices(
    rows: torch.Tensor,
    counts: torch.Tensor,
    selection_width: int,
    context_capacity: int,
    index_kind: str,
) -> torch.Tensor:
    """Packed [rows, selection_width + 1] indices with a trailing count column.

    index_kind="random": scattered tokens (legacy distribution).
    index_kind="group4": contiguous groups of 4 tokens, modelling QSA locality.
    """
    device = rows.device
    logical = torch.full(
        (rows.numel(), selection_width), -1, device=device, dtype=torch.int32
    )
    for r in range(rows.numel()):
        count = int(counts[r].item())
        if count <= 0:
            continue
        if index_kind == "random":
            perm = torch.randperm(context_capacity, device=device)[:count]
        elif index_kind == "group4":
            n_groups = (count + 3) // 4
            starts = torch.randint(0, context_capacity - 4, (n_groups,), device=device)
            perm = (
                starts[:, None] + torch.arange(4, device=device)[None, :]
            ).flatten()[:count]
        else:
            raise ValueError(f"unknown index_kind: {index_kind}")
        logical[r, :count] = perm.to(torch.int32)
    packed = torch.full(
        (rows.numel(), selection_width + 1), -1, device=device, dtype=torch.int32
    )
    packed[:, :selection_width] = logical
    packed[:, -1] = counts.to(torch.int32)
    return packed


@pytest.mark.qsa_sparse_paged_attention
@pytest.mark.parametrize(
    "num_rows, num_query_heads, num_kv_heads, page_size, "
    "use_prefill_config, num_requests, fp8",
    [
        # T, Hq, Hkv, page_size, prefill_config, requests, fp8
        (1, 24, 2, 1600, True, 2, False),
        (16, 12, 1, 1600, True, 3, False),
        (32, 6, 1, 800, True, 5, False),
        (128, 24, 2, 1568, True, 7, False),
        (257, 6, 1, 800, True, 13, False),
        (513, 6, 1, 784, True, 17, False),
        (700, 6, 1, 800, True, 23, False),
        (1024, 24, 2, 1600, True, 33, False),
        (2048, 24, 2, 1600, True, 63, False),
        (2048, 24, 2, 1600, False, 63, False),
        (1, 24, 2, 1600, True, 2, True),
        (128, 24, 2, 1568, True, 7, True),
        (2048, 24, 2, 1600, True, 63, True),
        (2048, 24, 2, 1600, False, 63, True),
    ],
)
@torch.inference_mode()
def test_qsa_sparse_paged_attention(
    num_rows: int,
    num_query_heads: int,
    num_kv_heads: int,
    page_size: int,
    use_prefill_config: bool,
    num_requests: int,
    fp8: bool,
) -> None:
    """Compare FlagGems with the vLLM op and its PyTorch reference."""
    if cfg.QUICK_MODE and num_rows not in (1, 16, 32):
        pytest.skip("outside the quick shape set")
    qsa = pytest.importorskip("vllm.models.qwen4_exp.nvidia.ops.qsa")
    torch.manual_seed(2)
    head_dim = 256
    num_pages_per_request = 65
    num_cache_blocks = num_requests * num_pages_per_request
    selection_width = 2051
    q = torch.randn(
        num_rows,
        num_query_heads,
        head_dim,
        device=flaggems_vllm.device,
        dtype=torch.bfloat16,
    )
    output_gate = torch.randn_like(q)
    kv_cache = torch.randn(
        num_cache_blocks,
        page_size,
        num_kv_heads,
        2 * head_dim,
        device=flaggems_vllm.device,
        dtype=torch.bfloat16,
    )
    k_cache, v_cache = kv_cache.split(head_dim, dim=-1)
    block_table = (
        torch.randperm(num_cache_blocks, device=flaggems_vllm.device)
        .reshape(num_requests, num_pages_per_request)
        .to(torch.int32)
    )
    rows = torch.arange(num_rows, device=flaggems_vllm.device, dtype=torch.int32)
    rows_per_request = (num_rows + num_requests - 1) // num_requests
    token_to_req = rows // rows_per_request

    # Distinct, scattered token indices per query; all fit in its request's cache.
    context_capacity = num_pages_per_request * page_size
    selected = torch.randperm(context_capacity, device=flaggems_vllm.device)[
        :selection_width
    ]
    selected = (selected[None, :] + rows[:, None]) % context_capacity
    counts = torch.where(rows % 3 == 1, 17, selection_width - rows % 4)
    ranks = torch.arange(selection_width, device=flaggems_vllm.device)
    logical_indices = torch.full(
        (num_rows, selection_width + 1),
        -1,
        device=flaggems_vllm.device,
        dtype=torch.int32,
    )
    logical_indices[:, :-1] = torch.where(ranks < counts[:, None], selected, -1)
    logical_indices[:, -1] = counts

    if fp8:
        # A fixed non-unit pair (k != v) exercises the host-side scale folding
        # and catches a k/v swap; scales are host floats, as the layer exposes
        # them. Stored values are the scaled ones, as reshape_and_cache does.
        k_scale, v_scale = 0.5, 2.0
        k_cache = (k_cache / k_scale).to(torch.float8_e4m3fn)
        v_cache = (v_cache / v_scale).to(torch.float8_e4m3fn)
    else:
        k_scale, v_scale = 1.0, 1.0

    expected = qsa.qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config=use_prefill_config,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=output_gate,
    )
    actual = flaggems_vllm.qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config=use_prefill_config,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=output_gate,
    )

    assert actual.shape == q.shape
    assert actual.dtype == q.dtype
    assert actual.device == q.device
    flaggems_vllm.testing.assert_close(actual, expected, dtype=q.dtype, atol=2e-2)

    torch_expected = _qsa_sparse_paged_attention_reference(
        q,
        k_cache,
        v_cache,
        logical_indices[:, :-1],  # Exclude the packed count column.
        block_table,
        token_to_req,
        head_dim**-0.5,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch_expected = torch_expected * torch.sigmoid(output_gate)
    flaggems_vllm.testing.assert_close(actual, torch_expected, dtype=q.dtype, atol=2e-2)


@pytest.mark.qsa_sparse_paged_attention
@pytest.mark.parametrize("index_kind", ["random", "group4"])
@pytest.mark.parametrize(
    "count_mode",
    [0, 1, 17, 31, 32, 33, 63, 64, 65, 2048, 2051, "short-long"],
)
@pytest.mark.parametrize("contiguous_kv", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
@torch.inference_mode()
def test_qsa_sparse_paged_attention_edge_indices(
    index_kind: str,
    count_mode,
    contiguous_kv: bool,
    fp8: bool,
) -> None:
    """Index distributions and lengths beyond the legacy regression set.

    Covers empty/short/tile-boundary selections, contiguous-token groups with
    QSA-like locality, and contiguous vs split (non-contiguous) K/V caches.
    """
    qsa = pytest.importorskip("vllm.models.qwen4_exp.nvidia.ops.qsa")
    torch.manual_seed(3)
    num_rows, num_query_heads, num_kv_heads = 32, 6, 1
    head_dim = 256
    page_size = 800
    num_requests = 5
    num_pages_per_request = 65
    num_cache_blocks = num_requests * num_pages_per_request
    selection_width = 2051
    q = torch.randn(
        num_rows,
        num_query_heads,
        head_dim,
        device=flaggems_vllm.device,
        dtype=torch.bfloat16,
    )
    output_gate = torch.randn_like(q)
    if contiguous_kv:
        k_cache = torch.randn(
            num_cache_blocks,
            page_size,
            num_kv_heads,
            head_dim,
            device=flaggems_vllm.device,
            dtype=torch.bfloat16,
        )
        v_cache = torch.randn_like(k_cache)
    else:
        kv_cache = torch.randn(
            num_cache_blocks,
            page_size,
            num_kv_heads,
            2 * head_dim,
            device=flaggems_vllm.device,
            dtype=torch.bfloat16,
        )
        k_cache, v_cache = kv_cache.split(head_dim, dim=-1)
    block_table = (
        torch.randperm(num_cache_blocks, device=flaggems_vllm.device)
        .reshape(num_requests, num_pages_per_request)
        .to(torch.int32)
    )
    rows = torch.arange(num_rows, device=flaggems_vllm.device, dtype=torch.int32)
    token_to_req = rows // ((num_rows + num_requests - 1) // num_requests)

    context_capacity = num_pages_per_request * page_size
    if count_mode == "short-long":
        counts = torch.where(rows % 2 == 0, 17, selection_width)
    else:
        counts = torch.full((num_rows,), count_mode, device=flaggems_vllm.device)
    logical_indices = _make_logical_indices(
        rows, counts, selection_width, context_capacity, index_kind
    )

    if fp8:
        k_scale, v_scale = 0.5, 2.0
        k_cache = (k_cache / k_scale).to(torch.float8_e4m3fn)
        v_cache = (v_cache / v_scale).to(torch.float8_e4m3fn)
    else:
        k_scale, v_scale = 1.0, 1.0

    expected = qsa.qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config=True,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=output_gate,
    )
    actual = flaggems_vllm.qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config=True,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=output_gate,
    )
    flaggems_vllm.testing.assert_close(actual, expected, dtype=q.dtype, atol=2e-2)

    torch_expected = _qsa_sparse_paged_attention_reference(
        q,
        k_cache,
        v_cache,
        logical_indices[:, :-1],
        block_table,
        token_to_req,
        head_dim**-0.5,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch_expected = torch_expected * torch.sigmoid(output_gate)
    flaggems_vllm.testing.assert_close(actual, torch_expected, dtype=q.dtype, atol=2e-2)


@pytest.mark.qsa_sparse_paged_attention
@torch.inference_mode()
def test_qsa_sparse_paged_attention_graph_replay_freshness() -> None:
    """Graph warmup + replay must not leak stale workspace into outputs.

    Capture the op on one input, then replay after changing the selection
    contents (same shapes): the result must equal a fresh eager run.
    """
    qsa = pytest.importorskip("vllm.models.qwen4_exp.nvidia.ops.qsa")
    torch.manual_seed(5)
    num_rows, num_query_heads, num_kv_heads = 32, 6, 1
    head_dim = 256
    page_size = 800
    num_requests = 5
    num_pages_per_request = 65
    num_cache_blocks = num_requests * num_pages_per_request
    selection_width = 2051

    def build(count_a: int, count_b: int, seed: int):
        g = torch.Generator(device=flaggems_vllm.device).manual_seed(seed)
        q = torch.randn(
            num_rows,
            num_query_heads,
            head_dim,
            device=flaggems_vllm.device,
            dtype=torch.bfloat16,
            generator=g,
        )
        output_gate = torch.randn(
            q.shape, device=flaggems_vllm.device, dtype=torch.bfloat16, generator=g
        )
        kv_cache = torch.randn(
            num_cache_blocks,
            page_size,
            num_kv_heads,
            2 * head_dim,
            device=flaggems_vllm.device,
            dtype=torch.bfloat16,
            generator=g,
        )
        k_cache, v_cache = kv_cache.split(head_dim, dim=-1)
        block_table = (
            torch.randperm(num_cache_blocks, device=flaggems_vllm.device, generator=g)
            .reshape(num_requests, num_pages_per_request)
            .to(torch.int32)
        )
        rows = torch.arange(num_rows, device=flaggems_vllm.device, dtype=torch.int32)
        token_to_req = rows // ((num_rows + num_requests - 1) // num_requests)
        counts = torch.where(rows % 2 == 0, count_a, count_b).to(torch.int32)
        logical_indices = _make_logical_indices(
            rows,
            counts,
            selection_width,
            num_pages_per_request * page_size,
            "random",
        )
        return (
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            output_gate,
        )

    q, k, v, li, bt, ttr, gate = build(17, 2048, 11)

    # Warm up on a side stream (required before capture), then capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flaggems_vllm.qsa_sparse_paged_attention(
                q,
                k,
                v,
                li,
                bt,
                ttr,
                use_prefill_config=True,
                output_gate=gate,
            )
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = flaggems_vllm.qsa_sparse_paged_attention(
            q,
            k,
            v,
            li,
            bt,
            ttr,
            use_prefill_config=True,
            output_gate=gate,
        )

    # Change the selection contents in place, keeping shapes identical.
    q2, k2, v2, li2, bt2, ttr2, gate2 = build(1, 2051, 23)
    q.copy_(q2)
    k.copy_(k2)
    v.copy_(v2)
    li.copy_(li2)
    bt.copy_(bt2)
    gate.copy_(gate2)
    graph.replay()

    eager = flaggems_vllm.qsa_sparse_paged_attention(
        q,
        k,
        v,
        li,
        bt,
        ttr,
        use_prefill_config=True,
        output_gate=gate,
    )
    flaggems_vllm.testing.assert_close(out, eager, dtype=q.dtype, atol=1e-3)

    # And the replayed result must still match the frozen vLLM op.
    vllm_out = qsa.qsa_sparse_paged_attention(
        q,
        k,
        v,
        li,
        bt,
        ttr,
        use_prefill_config=True,
        output_gate=gate,
    )
    flaggems_vllm.testing.assert_close(out, vllm_out, dtype=q.dtype, atol=2e-2)
