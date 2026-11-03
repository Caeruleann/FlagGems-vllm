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

import pytest

# import torch

# import flaggems_vllm

# from .conftest import QUICK_MODE

pytestmark = [pytest.mark.fused_deepseek_v4_kv_rope_insert]

# try:
#     import vllm._custom_ops  # noqa: F401 - registers torch.ops._C
# except (ImportError, AttributeError, RuntimeError, OSError):
#     pass
# VLLM_REF = getattr(torch.ops._C, OP_NAME, None)
