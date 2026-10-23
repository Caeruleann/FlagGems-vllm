# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark FlagGems QSA attention against the vLLM Triton implementation."""

import pytest
import torch

import flaggems_vllm

from . import base


def _make_packed_indices(num_rows, selection_width, context_capacity, device):
    """Packed [rows, selection_width + 1] indices with a trailing count column.

    Every row selects the full width: uniform random distinct indices over
    the context capacity (worst-case locality), and the count column is the
    selection width.
    """
    packed = torch.empty(
        (num_rows, selection_width + 1), device=device, dtype=torch.int32
    )
    for r in range(num_rows):
        perm = torch.randperm(context_capacity, device=device)[:selection_width]
        packed[r, :selection_width] = perm.to(torch.int32)
    packed[:, -1] = selection_width
    return packed


class QSASparsePagedAttentionBenchmark(base.Benchmark):
    def __init__(self, reference):
        super().__init__(
            "qsa_sparse_paged_attention",
            reference,
            [torch.bfloat16],
            gems_op=flaggems_vllm.qsa_sparse_paged_attention,
        )

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            (
                num_rows,
                num_query_heads,
                num_kv_heads,
                page_size,
                use_prefill_config,
                num_requests,
                fp8,
            ) = shape[:7]
            # Real-shape fidelity fields: pages per request and the physical
            # pool size (default: 65 pages and a fully-unique table, the
            # synthetic layout).
            num_pages_per_request = shape[7] if len(shape) > 7 else 65
            torch.manual_seed(2)
            head_dim = 256
            num_cache_blocks = (
                shape[8] if len(shape) > 8 else num_requests * num_pages_per_request
            )
            selection_width = 2051
            q = torch.randn(
                num_rows,
                num_query_heads,
                head_dim,
                device=flaggems_vllm.device,
                dtype=dtype,
            )
            output_gate = torch.randn_like(q)
            kv_cache = torch.randn(
                num_cache_blocks,
                page_size,
                num_kv_heads,
                2 * head_dim,
                device=flaggems_vllm.device,
                dtype=dtype,
            )
            k_cache, v_cache = kv_cache.split(head_dim, dim=-1)
            total_pages = num_requests * num_pages_per_request
            if num_cache_blocks >= total_pages:
                block_table = (
                    torch.randperm(num_cache_blocks, device=flaggems_vllm.device)[
                        :total_pages
                    ]
                    .reshape(num_requests, num_pages_per_request)
                    .to(torch.int32)
                )
            else:
                # Real pools are smaller than the logical table (not every page
                # is allocated); entries may repeat.
                block_table = torch.randint(
                    0,
                    num_cache_blocks,
                    (num_requests, num_pages_per_request),
                    device=flaggems_vllm.device,
                    dtype=torch.int32,
                )
            rows = torch.arange(
                num_rows, device=flaggems_vllm.device, dtype=torch.int32
            )
            rows_per_request = (num_rows + num_requests - 1) // num_requests
            token_to_req = rows // rows_per_request

            context_capacity = num_pages_per_request * page_size
            # Worst case: every row selects the full width, so no row hits the
            # short-row direct-write or empty-split paths.
            logical_indices = _make_packed_indices(
                num_rows,
                selection_width,
                context_capacity,
                flaggems_vllm.device,
            )

            if fp8:
                # A fixed non-unit pair (k != v) exercises the host-side scale folding
                # and catches a k/v swap; scales are host floats, as the layer exposes
                # them. Stored values are the scaled ones, as reshape_and_cache does.
                k_scale, v_scale = 0.5, 2.0
                # Quantize into the fused KV buffer and then split, so K/V keep
                # the production interleaved layout (views with a 2*head_dim
                # token stride) instead of becoming independent compact tensors.
                kv_fp8 = torch.empty_like(kv_cache, dtype=torch.float8_e4m3fn)
                k_cache, v_cache = kv_fp8.split(head_dim, dim=-1)
                k_cache.copy_(
                    (kv_cache[..., :head_dim] / k_scale).to(torch.float8_e4m3fn)
                )
                v_cache.copy_(
                    (kv_cache[..., head_dim:] / v_scale).to(torch.float8_e4m3fn)
                )
            else:
                k_scale, v_scale = 1.0, 1.0

            yield (
                q,
                k_cache,
                v_cache,
                logical_indices,
                block_table,
                token_to_req,
                {
                    "use_prefill_config": use_prefill_config,
                    "k_scale": k_scale,
                    "v_scale": v_scale,
                    "output_gate": output_gate,
                },
            )


@pytest.mark.qsa_sparse_paged_attention
@torch.inference_mode()
def test_qsa_sparse_paged_attention_benchmark():
    qsa = pytest.importorskip("vllm.models.qwen4_exp.nvidia.ops.qsa")
    QSASparsePagedAttentionBenchmark(qsa.qsa_sparse_paged_attention).run()
