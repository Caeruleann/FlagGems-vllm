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


def _make_packed_indices(rows, counts, selection_width, context_capacity, index_kind):
    """Packed [rows, selection_width + 1] indices with a trailing count column.

    index_kind="random": scattered tokens. index_kind="group4": contiguous
    groups of 4 tokens (QSA-like locality).
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
            index_kind = shape[7] if len(shape) > 7 else "random"
            count_mode = shape[8] if len(shape) > 8 else "mixed"
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
            block_table = (
                torch.randperm(num_cache_blocks, device=flaggems_vllm.device)
                .reshape(num_requests, num_pages_per_request)
                .to(torch.int32)
            )
            rows = torch.arange(
                num_rows, device=flaggems_vllm.device, dtype=torch.int32
            )
            rows_per_request = (num_rows + num_requests - 1) // num_requests
            token_to_req = rows // rows_per_request

            context_capacity = num_pages_per_request * page_size
            if count_mode == "mixed":
                # Legacy regression distribution: mostly long rows, every third
                # row short (17).
                counts = torch.where(rows % 3 == 1, 17, selection_width - rows % 4)
            elif count_mode == "short-long":
                counts = torch.where(rows % 2 == 0, 17, selection_width)
            elif count_mode == "short":
                counts = torch.full(
                    (num_rows,), 17, device=flaggems_vllm.device, dtype=torch.int32
                )
            elif count_mode == "long":
                counts = torch.full(
                    (num_rows,),
                    selection_width,
                    device=flaggems_vllm.device,
                    dtype=torch.int32,
                )
            else:
                raise ValueError(f"unknown count_mode: {count_mode}")
            logical_indices = _make_packed_indices(
                rows,
                counts.to(torch.int32),
                selection_width,
                context_capacity,
                index_kind,
            )

            if fp8:
                # A fixed non-unit pair (k != v) exercises the host-side scale folding
                # and catches a k/v swap; scales are host floats, as the layer exposes
                # them. Stored values are the scaled ones, as reshape_and_cache does.
                k_scale, v_scale = 0.5, 2.0
                k_cache = (k_cache / k_scale).to(torch.float8_e4m3fn)
                v_cache = (v_cache / v_scale).to(torch.float8_e4m3fn)
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
