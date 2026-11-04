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
import torch

import flaggems_vllm
from baseline.fused_deepseek_v4_kv_rope_insert import (
    CACHE_FORMATS,
    CORE_SHAPES,
    build_inputs,
    native,
)

from . import conftest as cfg

VLLM_REF = native()

pytestmark = pytest.mark.fused_deepseek_v4_kv_rope_insert

SHAPES = (
    [(1, 1, 32), (20, 17, 32), (20, 17, 256), (1026, 1023, 256)]
    if cfg.QUICK_MODE
    else CORE_SHAPES
)


def _make_inputs(
    num_tokens, num_tokens_insert, cache_block_size, cache_format, **input_kwargs
):
    binding = {
        "num_tokens": num_tokens,
        "num_tokens_insert": num_tokens_insert,
        "cache_block_size": cache_block_size,
        "cache_format": cache_format,
    }
    args, kwargs = build_inputs(
        binding, torch.bfloat16, flaggems_vllm.device, **input_kwargs
    )
    return list(args), kwargs


def _assert_cache_equal(actual, expected):
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )


@pytest.mark.skipif(VLLM_REF is None, reason="vLLM native op unavailable")
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@pytest.mark.parametrize("num_tokens,num_tokens_insert,cache_block_size", SHAPES)
@torch.inference_mode()
def test_matches_vllm_reference(
    num_tokens, num_tokens_insert, cache_block_size, cache_format
):
    args, kwargs = _make_inputs(
        num_tokens, num_tokens_insert, cache_block_size, cache_format
    )
    reference_args = list(args)
    reference_args[1] = args[1].clone()
    kv_before = args[0].clone()

    VLLM_REF(*reference_args, **kwargs)
    result = flaggems_vllm.fused_deepseek_v4_kv_rope_insert(*args, **kwargs)

    assert result is None
    torch.testing.assert_close(args[0], kv_before, rtol=0, atol=0)
    _assert_cache_equal(args[1], reference_args[1])


@pytest.mark.skipif(VLLM_REF is None, reason="vLLM native op unavailable")
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@torch.inference_mode()
def test_nontrivial_positions_slots_and_scale(cache_format):
    slots = torch.tensor(
        [9, 3, 11, -1, 0], dtype=torch.int64, device=flaggems_vllm.device
    )
    args, kwargs = _make_inputs(
        8,
        5,
        4,
        cache_format,
        slot_mapping=slots,
        position_offset=7,
        scale_value=0.7,
    )
    reference_args = list(args)
    reference_args[1] = args[1].clone()

    VLLM_REF(*reference_args, **kwargs)
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(*args, **kwargs)

    _assert_cache_equal(args[1], reference_args[1])


@pytest.mark.skipif(VLLM_REF is None, reason="vLLM native op unavailable")
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@torch.inference_mode()
def test_special_finite_values_match_vllm(cache_format):
    args, kwargs = _make_inputs(20, 17, 32, cache_format)
    args[0].zero_()
    args[0][0, 0::2] = -0.0
    args[0][1].fill_(1e10)
    args[0][2].fill_(1e-10)
    args[0][3].fill_(-1e10)
    args[0][4].fill_(-1e-10)
    reference_args = list(args)
    reference_args[1] = args[1].clone()

    VLLM_REF(*reference_args, **kwargs)
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(*args, **kwargs)

    torch.testing.assert_close(
        args[1].view(torch.uint8),
        reference_args[1].view(torch.uint8),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(VLLM_REF is None, reason="vLLM native op unavailable")
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@torch.inference_mode()
def test_matches_vllm_with_padded_pages(cache_format):
    cache_block_size = 256
    num_insert = 17
    num_blocks = (num_insert + cache_block_size - 1) // cache_block_size + 1
    slots = torch.randperm(
        num_blocks * cache_block_size, device=flaggems_vllm.device
    )[:num_insert]
    slots[::5] = -1
    args, kwargs = _make_inputs(
        20,
        num_insert,
        cache_block_size,
        cache_format,
        padded_pages=True,
        slot_mapping=slots,
    )
    page_stride = args[1].stride(0)
    backing = args[1].as_strided(
        (num_blocks, page_stride), (page_stride, 1)
    )
    reference_backing = backing.clone()
    reference_args = list(args)
    reference_args[1] = reference_backing.as_strided(args[1].shape, args[1].stride())
    kv_before = args[0].clone()

    VLLM_REF(*reference_args, **kwargs)
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(*args, **kwargs)

    torch.testing.assert_close(
        backing.view(torch.uint8),
        reference_backing.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(args[0], kv_before, rtol=0, atol=0)


@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@torch.inference_mode()
def test_empty_insert_is_noop(cache_format):
    args, kwargs = _make_inputs(4, 0, 32, cache_format)
    cache_before = args[1].clone()
    kv_before = args[0].clone()

    result = flaggems_vllm.fused_deepseek_v4_kv_rope_insert(*args, **kwargs)

    assert result is None
    torch.testing.assert_close(args[0], kv_before, rtol=0, atol=0)
    torch.testing.assert_close(args[1], cache_before, rtol=0, atol=0)


@torch.inference_mode()
def test_rejects_empty_cpu_input():
    binding = {
        "num_tokens": 0,
        "num_tokens_insert": 0,
        "cache_block_size": 32,
        "cache_format": "bf16",
    }
    args, kwargs = build_inputs(binding, torch.bfloat16, "cpu")

    with pytest.raises(NotImplementedError, match="requires CUDA tensors"):
        flaggems_vllm.fused_deepseek_v4_kv_rope_insert(*args, **kwargs)


@pytest.mark.skipif(
    VLLM_REF is None or torch.cuda.device_count() < 2,
    reason="non-current-device test needs vLLM and two CUDA devices",
)
@torch.inference_mode()
def test_launch_uses_kv_device():
    binding = {
        "num_tokens": 20,
        "num_tokens_insert": 17,
        "cache_block_size": 32,
        "cache_format": "bf16",
    }
    with torch.cuda.device(0):
        args, kwargs = build_inputs(binding, torch.bfloat16, "cuda:1")
        reference_args = list(args)
        reference_args[1] = args[1].clone()

        VLLM_REF(*reference_args, **kwargs)
        flaggems_vllm.fused_deepseek_v4_kv_rope_insert(*args, **kwargs)

        assert torch.cuda.current_device() == 0
        _assert_cache_equal(args[1], reference_args[1])


@torch.inference_mode()
def test_rejects_noncontiguous_kv():
    args, kwargs = _make_inputs(4, 4, 32, "bf16")
    args[0] = torch.randn(
        4, 1024, dtype=torch.bfloat16, device=flaggems_vllm.device
    )[:, ::2]

    with pytest.raises(RuntimeError, match="kv must be contiguous"):
        flaggems_vllm.fused_deepseek_v4_kv_rope_insert(*args, **kwargs)


@torch.inference_mode()
def test_rejects_unsupported_cache_dtype():
    args, kwargs = _make_inputs(4, 4, 32, "bf16")
    args[1] = torch.empty_like(args[1], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="cache must be"):
        flaggems_vllm.fused_deepseek_v4_kv_rope_insert(*args, **kwargs)


@torch.inference_mode()
def test_rejects_missing_fp8_scale():
    args, kwargs = _make_inputs(4, 4, 32, "fp8")
    args[6] = None

    with pytest.raises(RuntimeError, match="requires one float32 KV scale"):
        flaggems_vllm.fused_deepseek_v4_kv_rope_insert(*args, **kwargs)
