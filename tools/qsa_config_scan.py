# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

# QSA sparse paged attention optimization harness.
# Staged timing (main kernel / merge / whole op) + config scan support.
# Not part of the repo test suite; used for H800 tuning per opt.md.

from __future__ import annotations

import argparse
import sys

import torch
import triton

sys.path.insert(0, "src")

import flaggems_vllm  # noqa: F401,E402
from flaggems_vllm.ops.qsa_sparse_paged_attention import (  # noqa: E402
    _qsa_merge_splitk_kernel,
    _qsa_sparse_paged_gqa_splitk_kernel,
)

DEV = "cuda"
HEAD_DIM = 256
PAGES_PER_REQ = 65
SEL_WIDTH = 2051


def make_counts(rows, mode, sel_width=SEL_WIDTH):
    """Per-row valid counts. mode: int | 'mixed' | 'short-long'."""
    rows_t = torch.arange(rows, device=DEV)
    if mode == "mixed":
        # current test distribution: mostly long, every 3rd row short(17)
        return torch.where(rows_t % 3 == 1, 17, sel_width - rows_t % 4)
    if mode == "short-long":
        # half rows L=17, half rows full width
        return torch.where(rows_t % 2 == 0, 17, sel_width)
    if isinstance(mode, int):
        return torch.full((rows,), mode, device=DEV, dtype=torch.long)
    if mode.isdigit():
        return torch.full((rows,), int(mode), device=DEV, dtype=torch.long)
    raise ValueError(mode)


def make_indices(rows, counts, kind, pages_per_req=PAGES_PER_REQ, page_size=1600):
    """kind: 'random' scattered tokens, or 'group4' groups of 4 contiguous tokens."""
    sel_width = counts.max().item() if counts.numel() else 0
    sel_width = max(sel_width, 1)
    capacity = pages_per_req * page_size
    logical = torch.full((rows, sel_width), -1, device=DEV, dtype=torch.int32)
    for r in range(rows):
        c = int(counts[r].item())
        if c <= 0:
            continue
        if kind == "random":
            perm = torch.randperm(capacity, device=DEV)[:c]
        elif kind == "group4":
            n_groups = (c + 3) // 4
            starts = torch.randint(0, capacity - 4, (n_groups,), device=DEV)
            perm = (starts[:, None] + torch.arange(4, device=DEV)[None, :]).flatten()[
                :c
            ]
        else:
            raise ValueError(kind)
        logical[r, :c] = perm.to(torch.int32)
    out = torch.full((rows, sel_width + 1), -1, device=DEV, dtype=torch.int32)
    out[:, :sel_width] = logical
    out[:, -1] = counts.to(torch.int32)
    return out


def make_inputs(
    rows,
    hq,
    hkv,
    page_size,
    n_req,
    fp8=False,
    kind="random",
    count_mode="mixed",
    contiguous_kv=False,
    seed=2,
):
    torch.manual_seed(seed)
    n_blocks = n_req * PAGES_PER_REQ
    q = torch.randn(rows, hq, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    gate = torch.randn_like(q)
    if contiguous_kv:
        k_cache = torch.randn(
            n_blocks, page_size, hkv, HEAD_DIM, device=DEV, dtype=torch.bfloat16
        )
        v_cache = torch.randn(
            n_blocks, page_size, hkv, HEAD_DIM, device=DEV, dtype=torch.bfloat16
        )
    else:
        kv = torch.randn(
            n_blocks, page_size, hkv, 2 * HEAD_DIM, device=DEV, dtype=torch.bfloat16
        )
        k_cache, v_cache = kv.split(HEAD_DIM, dim=-1)
    block_table = (
        torch.randperm(n_blocks, device=DEV)
        .reshape(n_req, PAGES_PER_REQ)
        .to(torch.int32)
    )
    rows_t = torch.arange(rows, device=DEV, dtype=torch.int32)
    token_to_req = rows_t // ((rows + n_req - 1) // n_req)
    counts = make_counts(rows, count_mode)
    logical_indices = make_indices(rows, counts, kind, page_size=page_size)
    k_scale, v_scale = (0.5, 2.0) if fp8 else (1.0, 1.0)
    if fp8:
        k_cache = (k_cache / k_scale).to(torch.float8_e4m3fn)
        v_cache = (v_cache / v_scale).to(torch.float8_e4m3fn)
    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        logical_indices=logical_indices,
        block_table=block_table,
        token_to_req=token_to_req,
        use_prefill_config=True,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=gate,
    )


def time_fn(fn, warmup=5, iters=20):
    # Match the repo benchmark口径: triton.testing.do_bench.
    # warmup/rep are millisecond budgets (triton semantics).
    return triton.testing.do_bench(fn, warmup=warmup * 5, rep=iters * 5)


def launch_flaggems(inp, block_n, num_warps, num_stages, num_splits):
    """Mirror of the production host launch with explicit knobs; returns staged timers."""
    q = inp["q"]
    k_cache = inp["k_cache"]
    v_cache = inp["v_cache"]
    logical_indices = inp["logical_indices"]
    block_table = inp["block_table"]
    token_to_req = inp["token_to_req"]
    output_gate = inp["output_gate"]
    is_fp8 = k_cache.dtype == torch.float8_e4m3fn
    if is_fp8:
        softmax_scale = (q.shape[2] ** -0.5) * float(inp["k_scale"])
        output_scale = float(inp["v_scale"])
    else:
        softmax_scale = q.shape[2] ** -0.5
        output_scale = 1.0
    out = torch.empty_like(q)
    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    selection_width = logical_indices.shape[1] - 1
    num_tiles = triton.cdiv(selection_width, block_n)
    num_splits_eff = min(num_splits, num_tiles)
    if num_splits_eff == 1:
        partial_output = out
        partial_lse = out
        split_counts = out
    else:
        partial_output = torch.empty(
            (num_splits_eff, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits_eff, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )
        split_counts = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    grid = (q.shape[0], k_cache.shape[2], num_splits_eff)

    def main():
        _qsa_sparse_paged_gqa_splitk_kernel[grid](
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            partial_output,
            partial_lse,
            split_counts,
            out,
            softmax_scale,
            output_scale,
            output_gate,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            logical_indices.stride(0),
            block_table.stride(0),
            out.stride(0),
            out.stride(1),
            q.shape[0],
            k_cache.shape[0],
            block_table.shape[0],
            TOPK=selection_width,
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=q.shape[2],
            NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=num_splits_eff,
            T_BUCKET=triton.next_power_of_2(q.shape[0]),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            IS_FP8=is_fp8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def merge():
        _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
            partial_output,
            partial_lse,
            split_counts,
            out,
            output_gate,
            out.stride(0),
            out.stride(1),
            q.shape[0],
            HEAD_DIM=q.shape[2],
            NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=num_splits_eff,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits_eff),
            num_warps=2,
            num_stages=1,
        )

    def whole():
        main()
        if num_splits_eff > 1:
            merge()

    return whole, main, merge, out


def scan_config(inp, label, splits_list, block_list, warps_list, stages_list, args):
    vllm_mod = None
    if args.vllm_baseline:
        import vllm.models.qwen4_exp.nvidia.ops.qsa as vllm_mod

    def vllm_run():
        return vllm_mod.qsa_sparse_paged_attention(
            inp["q"],
            inp["k_cache"],
            inp["v_cache"],
            inp["logical_indices"],
            inp["block_table"],
            inp["token_to_req"],
            use_prefill_config=inp["use_prefill_config"],
            k_scale=inp["k_scale"],
            v_scale=inp["v_scale"],
            output_gate=inp["output_gate"],
        )

    base_ms = time_fn(vllm_run, args.warmup, args.iters) if vllm_mod else float("nan")
    print(f"\n=== {label} | vllm={base_ms:.4f} ms ===")
    print(
        f"{'BLOCK_N':>7} {'warps':>5} {'stages':>6} {'splits':>6}"
        f" {'main':>9} {'merge':>9} {'whole':>9} {'speedup':>8}"
    )
    results = []
    for splits in splits_list:
        for block_n in block_list:
            for warps in warps_list:
                for stages in stages_list:
                    if stages > 1 and block_n * max(1, warps) * 32 > 128 * 1024:
                        continue
                    try:
                        whole, main, merge, _ = launch_flaggems(
                            inp, block_n, warps, stages, splits
                        )
                        main_ms = time_fn(main, args.warmup, args.iters)
                        merge_ms = (
                            time_fn(merge, args.warmup, args.iters)
                            if splits > 1
                            else 0.0
                        )
                        whole_ms = time_fn(whole, args.warmup, args.iters)
                    except Exception as e:  # noqa: BLE001
                        print(
                            f"{block_n:>7} {warps:>5} {stages:>6} {splits:>6}  FAIL"
                            f" {type(e).__name__}: {str(e)[:60]}"
                        )
                        continue
                    sp = base_ms / whole_ms if vllm_mod else float("nan")
                    results.append((whole_ms, block_n, warps, stages, splits))
                    print(
                        f"{block_n:>7} {warps:>5} {stages:>6} {splits:>6} "
                        f"{main_ms:>9.4f} {merge_ms:>9.4f} {whole_ms:>9.4f} {sp:>8.3f}"
                    )
    results.sort()
    if results:
        w, b, wp, st, sp = results[0]
        print(
            f"--- best: BLOCK_N={b} warps={wp} stages={st} splits={sp} whole={w:.4f} ms"
        )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="scan", choices=["scan", "staged"])
    p.add_argument("--shape", default="1,24,2,1600,2")
    p.add_argument("--kind", default="random", choices=["random", "group4"])
    p.add_argument("--counts", default="mixed")
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--contig-kv", action="store_true")
    p.add_argument("--splits", default="")  # comma list
    p.add_argument("--blocks", default="")  # comma list
    p.add_argument("--warps", default="")  # comma list
    p.add_argument("--stages", default="")  # comma list
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--vllm-baseline", action="store_true")
    args = p.parse_args()

    rows, hq, hkv, page_size, n_req = [int(x) for x in args.shape.split(",")]
    inp = make_inputs(
        rows,
        hq,
        hkv,
        page_size,
        n_req,
        fp8=args.fp8,
        kind=args.kind,
        count_mode=args.counts,
        contiguous_kv=args.contig_kv,
    )
    label = (
        f"T={rows} Hq={hq} Hkv={hkv} ps={page_size} req={n_req}"
        f" kind={args.kind} counts={args.counts} fp8={args.fp8}"
    )

    splits = (
        [int(x) for x in args.splits.split(",")]
        if args.splits
        else [1, 2, 4, 8, 16, 32, 64]
    )
    blocks = [int(x) for x in args.blocks.split(",")] if args.blocks else [32, 64, 128]
    warps = [int(x) for x in args.warps.split(",")] if args.warps else [1, 2, 4, 8]
    stages = [int(x) for x in args.stages.split(",")] if args.stages else [1, 2, 3]
    scan_config(inp, label, splits, blocks, warps, stages, args)


if __name__ == "__main__":
    main()
