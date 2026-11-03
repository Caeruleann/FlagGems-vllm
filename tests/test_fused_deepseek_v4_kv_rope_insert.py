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


def _make_inputs(num_tokens, num_tokens_insert, cache_block_size, cache_format):
    binding = {
        "num_tokens": num_tokens,
        "num_tokens_insert": num_tokens_insert,
        "cache_block_size": cache_block_size,
        "cache_format": cache_format,
    }
    args, kwargs = build_inputs(binding, torch.bfloat16, flaggems_vllm.device)
    return list(args), kwargs


def _assert_cache_close(actual, expected, cache_format, cache_block_size):
    if cache_format == "bf16":
        torch.testing.assert_close(
            actual[..., :448], expected[..., :448], rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual[..., 448:], expected[..., 448:], rtol=0.008, atol=1e-6
        )
        return
    if cache_format == "fp8":
        torch.testing.assert_close(
            actual[..., :448], expected[..., :448], rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual[..., 448:].float(),
            expected[..., 448:].float(),
            rtol=0.13,
            atol=1e-2,
        )
        return

    actual_blocks = actual.reshape(actual.shape[0], -1)
    expected_blocks = expected.reshape(expected.shape[0], -1)
    packed_v4 = cache_format == "packed_v4"
    token_bytes = 576 if packed_v4 else 512
    scale_bytes = 8 if packed_v4 else 16
    token_data = actual_blocks[:, : cache_block_size * token_bytes].reshape(
        -1, cache_block_size, token_bytes
    )
    expected_token_data = expected_blocks[:, : cache_block_size * token_bytes].reshape(
        -1, cache_block_size, token_bytes
    )
    torch.testing.assert_close(
        token_data[..., :448], expected_token_data[..., :448], rtol=0, atol=0
    )
    if packed_v4:
        torch.testing.assert_close(
            token_data[..., 448:].view(torch.bfloat16),
            expected_token_data[..., 448:].view(torch.bfloat16),
            rtol=0.008,
            atol=1e-6,
        )
    else:
        torch.testing.assert_close(
            token_data[..., 448:].view(torch.float8_e4m3fn).float(),
            expected_token_data[..., 448:].view(torch.float8_e4m3fn).float(),
            rtol=0.13,
            atol=1e-2,
        )

    scale_start = cache_block_size * token_bytes
    scale_end = scale_start + cache_block_size * scale_bytes
    torch.testing.assert_close(
        actual_blocks[:, scale_start:scale_end],
        expected_blocks[:, scale_start:scale_end],
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(VLLM_REF is None, reason="vLLM native op unavailable")
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@torch.inference_mode()
def test_vllm_baseline_smoke(cache_format):
    args, kwargs = _make_inputs(4, 4, 32, cache_format)
    cache_before = args[1].clone()
    kv_before = args[0].clone()

    result = VLLM_REF(*args, **kwargs)

    assert result is None
    assert not torch.equal(args[1], cache_before)
    torch.testing.assert_close(args[0], kv_before, rtol=0, atol=0)


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

    VLLM_REF(*reference_args, **kwargs)
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(*args, **kwargs)

    _assert_cache_close(args[1], reference_args[1], cache_format, cache_block_size)


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
    args, kwargs = _make_inputs(20, 17, cache_block_size, cache_format)
    width = args[1].shape[2]
    page_stride = ((cache_block_size * width + 575) // 576) * 576 + 576
    backing = torch.full(
        (args[1].shape[0], page_stride), 3.0, device=flaggems_vllm.device
    ).to(args[1].dtype)
    args[1] = backing.as_strided(args[1].shape, (page_stride, width, 1))
    args[2] = torch.randperm(
        args[1].shape[0] * cache_block_size, device=flaggems_vllm.device
    )[:17]
    args[2][::5] = -1

    reference_backing = backing.clone()
    reference_args = list(args)
    reference_args[1] = reference_backing.as_strided(args[1].shape, args[1].stride())
    kv_before = args[0].clone()

    VLLM_REF(*reference_args, **kwargs)
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(*args, **kwargs)

    _assert_cache_close(args[1], reference_args[1], cache_format, cache_block_size)
    page_data_size = cache_block_size * width
    torch.testing.assert_close(
        backing[:, page_data_size:],
        reference_backing[:, page_data_size:],
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
