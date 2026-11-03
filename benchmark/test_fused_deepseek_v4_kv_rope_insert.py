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

"""Native baseline scaffold; no fake FlagGems latency or speedup is reported.

Run the baseline with:
    pytest -s benchmark/test_fused_deepseek_v4_kv_rope_insert.py -k baseline \
        --level core --iter 1 --warmup 1

Remove the comparison test's skip marker once the Triton kernels are ready.
The common harness reports median kernel latency in ms and base/gems speedup.
"""

import pytest

# import torch

# import flaggems_vllm

# from baseline.fused_deepseek_v4_kv_rope_insert import (
#     CACHE_FORMATS,
#     OP_NAME,
#     build_inputs,
#     native,
# )

# from . import base

# VLLM_REF = native()

pytestmark = pytest.mark.fused_deepseek_v4_kv_rope_insert


# class FusedDeepseekV4KVRopeInsertBenchmark(base.Benchmark):
#     def __init__(self, cache_format, baseline_only=False):
#         super().__init__(OP_NAME, VLLM_REF, [torch.bfloat16], is_inplace=True)
#         self.cache_format = cache_format
#         self.set_gems(flaggems_vllm.fused_deepseek_v4_kv_rope_insert)
