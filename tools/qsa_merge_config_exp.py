# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Merge kernel configuration experiment for QSA small-T cases.

Evidence follow-up per review feedback: the merge regression analysis must
separate register spills from the split_counts load dependency, and the
host-cost claims need single-variable comparisons. This script ONLY times
kernel variants; it does not modify production code.

Arms per case (fixed production main launch, fixed workspace buffers):
  - production merge (split_counts mask + early return) at num_warps 1/2/4
  - no-count clone (compile-time NUM_SPLITS mask, no split_counts read, no
    early return) at num_warps 1/2/4 -- timing only: stale workspace slots
    are zero-filled once before the loop and never written by either kernel.

Metrics per arm: n_regs, n_spills, merge-only do_bench median, and whole
chain (main+merge) do_bench_cudagraph median.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "src"), str(ROOT / "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from benchmark.test_qsa_sparse_paged_attention import (  # noqa: E402
    QSASparsePagedAttentionBenchmark,
)
from qsa_small_t_debug import SHAPES, observe_entries  # noqa: E402


@triton.jit(do_not_specialize=["num_rows"])
def _merge_no_count_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    """vLLM-style merge: compile-time split mask, no split_counts, no early
    return. Timing-only variant used to isolate the split_counts dependency."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    stride_output_gate_row = NUM_QUERY_HEADS * HEAD_DIM
    stride_output_gate_head = HEAD_DIM
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    split_rows = split_offsets.to(tl.int64) * num_rows + row
    partial_output = tl.load(
        partial_output_ptr
        + (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    merged = merged.to(output_ptr.dtype.element_ty)
    output_gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head * stride_output_gate_head
        + dim_offsets
    ).to(tl.float32)
    merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


def _build_case(num_rows: int, kv_dtype: str):
    import flaggems_vllm

    hq, hkv, page_size, requests = SHAPES[num_rows]
    shape = [
        num_rows,
        hq,
        hkv,
        page_size,
        True,
        requests,
        kv_dtype == "fp8",
        "random",
        "mixed",
    ]
    bench = QSASparsePagedAttentionBenchmark(flaggems_vllm.qsa_sparse_paged_attention)
    bench.shapes = [shape]
    packed = next(bench.get_input_iter(torch.bfloat16))
    return packed[:-1], packed[-1]


def _probe_production(inputs, kwargs):
    """Run the production op once and capture the exact main/merge launches."""
    import flaggems_vllm

    gems = importlib.import_module("flaggems_vllm.ops.qsa_sparse_paged_attention")
    entries = {
        "main_entry": gems._QSA_MAIN_TUNED,
        "merge_entry": gems._QSA_MERGE_TUNED,
    }
    snapshots = {}
    with observe_entries(entries, snapshots):
        out = flaggems_vllm.qsa_sparse_paged_attention(*inputs, **kwargs)
    torch.cuda.synchronize()
    main = snapshots["main_entry"][-1]
    merge = snapshots["merge_entry"][-1]
    return main, merge, out


def _reg_spills(kernel):
    return getattr(kernel, "n_regs", None), getattr(kernel, "n_spills", None)


def run_case(num_rows: int, kv_dtype: str, rep_ms: float, warmup_ms: float):
    gems = importlib.import_module("flaggems_vllm.ops.qsa_sparse_paged_attention")
    inputs, kwargs = _build_case(num_rows, kv_dtype)
    q = inputs[0]
    main_rec, merge_rec, out = _probe_production(inputs, kwargs)
    mc = main_rec["constexprs"]
    gc = merge_rec["constexprs"]
    num_rows_eff = q.shape[0]
    num_splits = int(mc["NUM_SPLITS"])
    # Rebuild the exact production main launch.
    grid_main = tuple(main_rec["grid"][:3])
    po = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
    pl = torch.empty(
        (num_splits, q.shape[0], q.shape[1]),
        dtype=torch.float32,
        device=q.device,
    )
    sc = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if out is None:
        out = torch.empty_like(q)
    gate_view = kwargs["output_gate"].view_as(q)
    if kv_dtype == "fp8":
        softmax_scale = (q.shape[2] ** -0.5) * float(kwargs["k_scale"])
        output_scale = float(kwargs["v_scale"])
    else:
        softmax_scale = q.shape[2] ** -0.5
        output_scale = 1.0
    main_kwargs = dict(
        TOPK=mc["TOPK"],
        PAGE_SIZE=mc["PAGE_SIZE"],
        PAGE_TABLE_WIDTH=mc["PAGE_TABLE_WIDTH"],
        GROUP_SIZE=mc["GROUP_SIZE"],
        HEAD_DIM=mc["HEAD_DIM"],
        NUM_QUERY_HEADS=mc["NUM_QUERY_HEADS"],
        NUM_SPLITS=num_splits,
        T_BUCKET=mc["T_BUCKET"],
        BLOCK_M=mc["BLOCK_M"],
        BLOCK_N=mc["BLOCK_N"],
        IS_FP8=mc["IS_FP8"],
        num_warps=main_rec["num_warps_effective"],
        num_stages=mc.get("num_stages", 2),
    )
    main_args = (
        q,
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        po,
        pl,
        sc,
        out,
        softmax_scale,
        output_scale,
        gate_view,
        q.stride(0),
        q.stride(1),
        inputs[1].stride(0),
        inputs[1].stride(1),
        inputs[1].stride(2),
        inputs[2].stride(0),
        inputs[2].stride(1),
        inputs[2].stride(2),
        inputs[3].stride(0),
        inputs[4].stride(0),
        out.stride(0),
        out.stride(1),
        num_rows_eff,
        inputs[1].shape[0],
        inputs[4].shape[0],
    )

    def main_launch():
        return gems._qsa_sparse_paged_gqa_splitk_kernel[grid_main](
            *main_args, **main_kwargs
        )

    # Merge launchers: production kernel and the no-count clone.
    grid_merge = tuple(merge_rec["grid"][:3])
    block_splits = int(gc["BLOCK_SPLITS"])

    def merge_prod(warps):
        return lambda: gems._qsa_merge_splitk_kernel[grid_merge](
            po,
            pl,
            sc,
            out,
            gate_view,
            out.stride(0),
            out.stride(1),
            num_rows_eff,
            HEAD_DIM=q.shape[2],
            NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=num_splits,
            BLOCK_SPLITS=block_splits,
            num_warps=warps,
            num_stages=1,
        )

    def merge_clone(warps):
        return lambda: _merge_no_count_kernel[grid_merge](
            po,
            pl,
            out,
            gate_view,
            out.stride(0),
            out.stride(1),
            num_rows_eff,
            HEAD_DIM=q.shape[2],
            NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=num_splits,
            BLOCK_SPLITS=block_splits,
            num_warps=warps,
            num_stages=1,
        )

    # Third arm: the vLLM wheel merge kernel itself, at its own S (capped at
    # the production split count) on the same buffers. Stale slots are
    # zero-filled once; timing only.
    import vllm.models.qwen4_exp.nvidia.ops.qsa as vllm_qsa

    vllm_num_splits = min(33, num_splits)

    def merge_vllm(warps):
        return lambda: vllm_qsa._qsa_merge_splitk_kernel[grid_merge](
            po,
            pl,
            out,
            gate_view,
            out.stride(0),
            out.stride(1),
            gate_view.stride(0),
            gate_view.stride(1),
            num_rows_eff,
            HEAD_DIM=q.shape[2],
            NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=vllm_num_splits,
            BLOCK_SPLITS=triton.next_power_of_2(vllm_num_splits),
            num_warps=warps,
            num_stages=1,
        )

    def merge_vllm_s64(warps):
        return lambda: vllm_qsa._qsa_merge_splitk_kernel[grid_merge](
            po,
            pl,
            out,
            gate_view,
            out.stride(0),
            out.stride(1),
            gate_view.stride(0),
            gate_view.stride(1),
            num_rows_eff,
            HEAD_DIM=q.shape[2],
            NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=num_splits,
            BLOCK_SPLITS=block_splits,
            num_warps=warps,
            num_stages=1,
        )

    results = {
        "case": f"t{num_rows}_{kv_dtype}",
        "num_splits": num_splits,
        "block_splits": block_splits,
        "main": {
            "grid": list(grid_main),
            "BLOCK_N": mc["BLOCK_N"],
            "num_warps": main_rec["num_warps_effective"],
            "num_stages": mc.get("num_stages"),
            "n_regs": main_rec["n_regs"],
            "n_spills": main_rec["n_spills"],
        },
        "arms": [],
    }

    first_kernel = main_launch()
    torch.cuda.synchronize()
    results["main"]["n_regs"], results["main"]["n_spills"] = _reg_spills(first_kernel)

    for arm_name, factory in (
        ("prod", merge_prod),
        ("no_count", merge_clone),
        ("vllm", merge_vllm),
        ("vllm_s64", merge_vllm_s64),
    ):
        for warps in (1, 2, 4):
            po.zero_()
            pl.zero_()
            sc.zero_()
            merge_fn = factory(warps)
            main_launch()
            kernel = merge_fn()
            torch.cuda.synchronize()
            regs, spills = _reg_spills(kernel)
            merge_ms = triton.testing.do_bench(
                merge_fn, warmup=warmup_ms, rep=rep_ms, return_mode="median"
            )

            def chain():
                main_launch()
                merge_fn()

            for _ in range(3):
                chain()
            torch.cuda.synchronize()
            chain_ms = triton.testing.do_bench_cudagraph(
                chain, rep=rep_ms, return_mode="median"
            )
            results["arms"].append(
                {
                    "arm": arm_name,
                    "num_warps": warps,
                    "n_regs": regs,
                    "n_spills": spills,
                    "merge_median_us": merge_ms * 1000,
                    "chain_cudagraph_us": chain_ms * 1000,
                }
            )
            print(
                f"t{num_rows}_{kv_dtype} {arm_name} w{warps}: "
                f"regs={regs} spills={spills} merge={merge_ms * 1000:.2f}us "
                f"chain={chain_ms * 1000:.2f}us",
                flush=True,
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[32, 1])
    parser.add_argument("--kv-dtypes", nargs="+", default=["bf16", "fp8"])
    parser.add_argument("--rep-ms", type=float, default=20)
    parser.add_argument("--warmup-ms", type=float, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.inference_mode().__enter__()
    all_results = []
    for num_rows in args.rows:
        for kv_dtype in args.kv_dtypes:
            all_results.append(
                run_case(num_rows, kv_dtype, args.rep_ms, args.warmup_ms)
            )
    if args.out:
        args.out.write_text(json.dumps(all_results, indent=2) + "\n")
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
