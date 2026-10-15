# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Regression tests for the LibEntry launch hot path.

These guard the minimal, behavior-preserving launch optimizations used by
decode-latency-sensitive ops: cached parameter metadata, frozenset membership
tests, the positional fast path for assembling all_args, and key construction
(dtype/alignment dispatch, constexpr bucketing).
"""

import pytest
import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry, libtuner


@triton.jit(do_not_specialize=["n"])
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


@triton.jit(do_not_specialize=["n"])
def _bucket_kernel(x_ptr, out_ptr, n, BUCKET: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + BUCKET, mask=mask)


@triton.jit
def _defaults_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, SCALE: tl.constexpr = 3):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a * SCALE, mask=mask)


def _tuned_add():
    return libentry()(
        libtuner(
            configs=[
                triton.Config(kwargs={"BLOCK": 64}, num_warps=4),
                triton.Config(kwargs={"BLOCK": 128}, num_warps=4),
            ],
            key=["n"],
        )(_add_kernel)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_libentry_cache_hit_with_fresh_tensors():
    """Repeated calls with new tensor instances of the same signature must
    produce correct results through the outer-cache hit path."""
    kernel = _tuned_add()
    for _ in range(3):
        n = 1000
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        out = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]), 1, 1)
        kernel[grid](x, y, out, n)
        torch.cuda.synchronize()
        assert torch.equal(out, x + y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_libentry_dispatch_by_dtype_and_bucket():
    """dtype and constexpr bucket changes must reach distinct cache entries
    (the T_BUCKET fix relies on this)."""
    bucketed = libentry()(
        libtuner(
            configs=[triton.Config(kwargs={"BLOCK": 64}, num_warps=4)],
            key=["BUCKET"],
        )(_bucket_kernel)
    )
    n = 256
    x = torch.randn(n, device="cuda")
    for bucket in (64, 128):
        out = torch.empty_like(x)
        grid = (triton.cdiv(n, 64), 1, 1)
        bucketed[grid](x, out, n, BUCKET=bucket)
        torch.cuda.synchronize()
        assert torch.equal(out, x + bucket)
    # Repeated call with the first bucket must still resolve the right binary.
    out = torch.empty_like(x)
    bucketed[grid](x, out, n, BUCKET=64)
    torch.cuda.synchronize()
    assert torch.equal(out, x + 64)

    # Different tensor dtypes must not share a specialization.
    xf = torch.randn(n, device="cuda", dtype=torch.float64)
    of = torch.empty_like(xf)
    bucketed[grid](xf, of, n, BUCKET=64)
    torch.cuda.synchronize()
    assert torch.equal(of, xf + 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_libentry_positional_args_and_default():
    """Positional-argument fast path (all_args = list(args) + trailing) plus
    constexpr defaults resolve to the same values as the generic path."""
    wrapped = libentry()(_defaults_kernel)
    n = 128
    x = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    grid = (1, 1, 1)
    # All positional.
    wrapped[grid](x, out, n, 128)
    torch.cuda.synchronize()
    assert torch.equal(out, x * 3)
    # Keyword constexpr form.
    wrapped[grid](x, out, n, BLOCK=128, SCALE=5)
    torch.cuda.synchronize()
    assert torch.equal(out, x * 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_libentry_tuple_and_lambda_grid():
    kernel = libentry()(
        libtuner(
            configs=[triton.Config(kwargs={"BLOCK": 128}, num_warps=4)],
            key=["n"],
        )(_add_kernel)
    )
    n = 512
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out1 = torch.empty_like(x)
    out2 = torch.empty_like(x)
    kernel[(triton.cdiv(n, 128), 1, 1)](x, y, out1, n)
    kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]), 1, 1)](x, y, out2, n)
    torch.cuda.synchronize()
    assert torch.equal(out1, x + y)
    assert torch.equal(out2, x + y)
