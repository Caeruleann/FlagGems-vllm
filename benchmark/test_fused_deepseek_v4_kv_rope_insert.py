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

"""Benchmark FlagGems-vllm against the native vLLM CUDA kernel.

Run only the native baseline with:
    pytest -s benchmark/test_fused_deepseek_v4_kv_rope_insert.py \
        --level core --iter 1 --warmup 1 --metrics latency_base

Run the CUDA Graph replay comparison on both implementations with:
    pytest -s benchmark/test_fused_deepseek_v4_kv_rope_insert.py \
        --mode cudagraph --level core --iter 100

The comparison test reports native latency, Triton latency, and speedup. If the
native op is unavailable, it is skipped and the separate Triton-only test
reports latency without claiming a native comparison.
"""

import pytest
import torch

import flaggems_vllm
from baseline.fused_deepseek_v4_kv_rope_insert import (
    CACHE_FORMATS,
    EXTENDED_SHAPES,
    OP_NAME,
    build_inputs,
    native,
)

from . import base

VLLM_REF = native()

pytestmark = pytest.mark.fused_deepseek_v4_kv_rope_insert


class FusedDeepseekV4KVRopeInsertBenchmark(base.Benchmark):
    def __init__(self, cache_format):
        super().__init__(OP_NAME, VLLM_REF, [torch.bfloat16], is_inplace=True)
        self.cache_format = cache_format
        self.set_gems(flaggems_vllm.fused_deepseek_v4_kv_rope_insert)

    def set_more_shapes(self):
        return EXTENDED_SHAPES

    def get_input_iter(self, dtype):
        for num_tokens, num_tokens_insert, cache_block_size in self.shapes:
            binding = {
                "num_tokens": num_tokens,
                "num_tokens_insert": num_tokens_insert,
                "cache_block_size": cache_block_size,
                "cache_format": self.cache_format,
            }
            args, _ = build_inputs(binding, dtype, self.device)
            yield args


@pytest.mark.skipif(VLLM_REF is None, reason="vLLM native comparison unavailable")
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
def test_fused_deepseek_v4_kv_rope_insert(cache_format):
    bench = FusedDeepseekV4KVRopeInsertBenchmark(cache_format)
    bench.run()


@pytest.mark.skipif(VLLM_REF is not None, reason="native comparison is available")
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
def test_fused_deepseek_v4_kv_rope_insert_triton_only(cache_format):
    bench = FusedDeepseekV4KVRopeInsertBenchmark(cache_format)
    bench.metrics = ["latency"]
    bench.run()
