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

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def fused_deepseek_v4_kv_rope_insert(
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache_block_size: int,
    fp8_scale: Optional[torch.Tensor] = None,
    kv_mxfp8: bool = False,
) -> None:
    """Apply GPT-J RoPE to KV's last 64 dims and insert into a paged cache.

    Mirrors ``torch.ops._C.fused_deepseek_v4_kv_rope_insert`` in vLLM's
    ``csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu``.
    ``kv`` is read-only contiguous BF16 [tokens, 512]. Only ``k_cache`` is
    mutated; negative slots and rows omitted by ``slot_mapping`` are skipped.
    Positions must index valid rows of the FP32 cos||sin cache [max_pos, 64].
    Live slots must be unique and within the cache allocation.

    Cache formats [num_blocks, cache_block_size, width]:
      * BF16 / E4M3: width 512; E4M3 requires a scalar FP32 ``fp8_scale``.
      * uint8: width 584 (V4) or 528 (MXFP8, ``kv_mxfp8=True``).
        Packed blocks store token data followed by scales, rather than
        independent token records; only the block stride is used.

    This is a host-wrapper scaffold. Empty insertions return immediately;
    nonempty insertions raise until the Triton kernels are implemented.
    No autotune configuration is needed until those kernels exist.
    """
    logger.debug("GEMS FUSED_DEEPSEEK_V4_KV_ROPE_INSERT")
    if not (
        kv.is_contiguous()
        and kv.ndim == 2
        and kv.shape[1] == 512
        and kv.dtype == torch.bfloat16
    ):
        raise RuntimeError("kv must be contiguous bfloat16 [tokens, 512]")
    if not (
        k_cache.device == kv.device
        and k_cache.ndim == 3
        and cache_block_size > 0
        and k_cache.shape[1] == cache_block_size
        and k_cache.stride(2) == 1
    ):
        raise RuntimeError("k_cache must be a paged cache on the KV device")
    for tensor in (slot_mapping, position_ids):
        if not (
            tensor.device == kv.device
            and tensor.dtype == torch.int64
            and tensor.ndim == 1
            and tensor.is_contiguous()
        ):
            raise RuntimeError(
                "slots and positions must be contiguous int64 on the KV device"
            )
    if not (
        position_ids.shape[0] == kv.shape[0] and slot_mapping.shape[0] <= kv.shape[0]
    ):
        raise RuntimeError("positions must match KV rows; slots may omit padded rows")
    if not (
        cos_sin_cache.device == kv.device
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.ndim == 2
        and cos_sin_cache.shape[1] == 64
        and cos_sin_cache.is_contiguous()
    ):
        raise RuntimeError("cos_sin_cache must be contiguous float32 [max_pos, 64]")
    packed = k_cache.dtype == torch.uint8
    fp8 = k_cache.dtype == torch.float8_e4m3fn
    if not (packed or fp8 or k_cache.dtype == torch.bfloat16):
        raise RuntimeError("cache must be uint8, bfloat16 or float8_e4m3fn")
    cache_width = (528 if kv_mxfp8 else 584) if packed else 512
    if k_cache.shape[2] != cache_width:
        raise RuntimeError("cache row width does not match its dtype")
    if fp8 and not (
        fp8_scale is not None
        and fp8_scale.device == kv.device
        and fp8_scale.dtype == torch.float32
        and fp8_scale.numel() == 1
    ):
        raise RuntimeError("fp8 cache requires one float32 KV scale on the KV device")
    if slot_mapping.shape[0] == 0:
        return

    raise NotImplementedError(
        "fused_deepseek_v4_kv_rope_insert Triton kernels are not implemented"
    )
