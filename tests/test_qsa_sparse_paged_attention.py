# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors

"""QSA attention porting skeleton, using vLLM v0.31.0/main as the baseline.

Adapted from vLLM tests/models/qwen4_exp/test_qsa_reference.py at f47792af7f.
The target currently raises NotImplementedError until its kernel is implemented.
Inputs use synthetic token indices directly, without indexer or block expansion.
The baseline already applies the output gate.
"""

import pytest
import torch

import flaggems_vllm

from . import conftest as cfg

pytestmark = pytest.mark.qsa_sparse_paged_attention


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
    """Compare the future FlagGems implementation with the vLLM op."""
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
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
