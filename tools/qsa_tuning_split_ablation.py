"""Ablation: pre-optimization op (bc8a9ea) + ONLY H800 configs & split policy.

Three arms per shape, same session, kernel-mode timing (do_bench median,
warmup=1000ms, rep=100ms — the acceptance defaults):
  vllm  — vLLM reference op
  abl   — old kernel structure (no empty-split skip, merge reads all splits,
          no split_counts, plain jit launch) with per-shape swept
          (BLOCK_N, warps, stages); NUM_SPLITS from the H800 policy table
          clamped at the 64-wide tile count (cap64) and at the chosen
          BLOCK_N tile count (old host invariant)
  prod  — current production op (full optimizations)
"""

import importlib.util
import json
import math
import subprocess
import tempfile
from pathlib import Path

import torch
import triton
from vllm.models.qwen4_exp.nvidia.ops import qsa as vllm_qsa

import flaggems_vllm

_old_src = subprocess.run(
    ["git", "show", "bc8a9ea:src/flaggems_vllm/ops/qsa_sparse_paged_attention.py"],
    capture_output=True,
    text=True,
    check=True,
    cwd=Path(__file__).resolve().parent.parent,
).stdout
# The pre-optimization host hardcodes num_stages=2 in the main launch; make it
# injectable for the sweep (module global resolved at call time).
_old_src = _old_src.replace(
    "        num_stages=2,", "        num_stages=_STAGES_OVERRIDE,", 1
)
_tmp = Path(tempfile.mkdtemp(prefix="qsa_ablation_")) / "old_op.py"
_tmp.write_text(_old_src)
spec = importlib.util.spec_from_file_location("qsa_old", _tmp)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

WIDTH = 2051

SHAPES = [
    (1, 24, 2, 1600, True, 2, False),
    (16, 12, 1, 1600, True, 3, False),
    (32, 6, 1, 800, True, 5, False),
    (128, 24, 2, 1568, True, 7, False),
    (257, 6, 1, 800, True, 13, False),
    (513, 6, 1, 784, True, 17, False),
    (700, 6, 1, 800, True, 23, False),
    (1024, 24, 2, 1600, True, 33, False),
    (2048, 24, 2, 1600, True, 63, False),
    (2048, 24, 2, 1600, False, 63, False),
    (1, 24, 2, 1600, True, 2, True),
    (128, 24, 2, 1568, True, 7, True),
    (2048, 24, 2, 1600, True, 63, True),
    (2048, 24, 2, 1600, False, 63, True),
    (128, 24, 2, 1568, True, 7, False, "group4", "mixed"),
    (128, 24, 2, 1568, True, 7, False, "random", "short"),
    (128, 24, 2, 1568, True, 7, False, "random", "long"),
    (128, 24, 2, 1568, True, 7, False, "group4", "short-long"),
    (2048, 24, 2, 1600, True, 63, False, "group4", "short-long"),
]

BF16_TABLE = ((32, 64), (256, 2), (512, 4), (1 << 62, 1))
FP8_TABLE = ((24, 64), (32, 16), (64, 8), (128, 4), (256, 8), (2048, 4), (1 << 62, 1))


def h800_target(rows, hkv, fp8):
    bp = rows * hkv
    for ub, s in (FP8_TABLE if fp8 else BF16_TABLE):
        if bp <= ub:
            return s
    return 1


def _make_packed_indices(rows, counts, selection_width, context_capacity, index_kind):
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
            raise ValueError(index_kind)
        logical[r, :count] = perm.to(torch.int32)
    packed = torch.full(
        (rows.numel(), selection_width + 1), -1, device=device, dtype=torch.int32
    )
    packed[:, :selection_width] = logical
    packed[:, -1] = counts.to(torch.int32)
    return packed


def make_inputs(shape):
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
        num_rows, num_query_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    output_gate = torch.randn_like(q)
    kv_cache = torch.randn(
        num_cache_blocks,
        page_size,
        num_kv_heads,
        2 * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache, v_cache = kv_cache.split(head_dim, dim=-1)
    block_table = (
        torch.randperm(num_cache_blocks, device="cuda")
        .reshape(num_requests, num_pages_per_request)
        .to(torch.int32)
    )
    rows = torch.arange(num_rows, device="cuda", dtype=torch.int32)
    rows_per_request = (num_rows + num_requests - 1) // num_requests
    token_to_req = rows // rows_per_request
    context_capacity = num_pages_per_request * page_size
    if count_mode == "mixed":
        counts = torch.where(rows % 3 == 1, 17, selection_width - rows % 4)
    elif count_mode == "short-long":
        counts = torch.where(rows % 2 == 0, 17, selection_width)
    elif count_mode == "short":
        counts = torch.full((num_rows,), 17, device="cuda", dtype=torch.int32)
    elif count_mode == "long":
        counts = torch.full(
            (num_rows,), selection_width, device="cuda", dtype=torch.int32
        )
    else:
        raise ValueError(count_mode)
    logical_indices = _make_packed_indices(
        rows, counts.to(torch.int32), selection_width, context_capacity, index_kind
    )
    if fp8:
        k_scale, v_scale = 0.5, 2.0
        k_cache = (k_cache / k_scale).to(torch.float8_e4m3fn)
        v_cache = (v_cache / v_scale).to(torch.float8_e4m3fn)
    else:
        k_scale, v_scale = 1.0, 1.0
    args = (q, k_cache, v_cache, logical_indices, block_table, token_to_req)
    kwargs = {
        "use_prefill_config": use_prefill_config,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "output_gate": output_gate,
    }
    return args, kwargs


def bench(fn, warmup, rep):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def shape_name(shape):
    t, pre, fp8 = shape[0], int(shape[4]), shape[6]
    tag = "fp8" if fp8 else "bf16"
    extra = ""
    if len(shape) > 7:
        extra = f"/{shape[7]}-{shape[8]}"
    return f"{tag} T={t} pre={pre}{extra}"


CONFIGS = [(bn, w, st) for bn in (32, 64, 128) for w in (2, 4, 8) for st in (1, 2, 3)]


def make_old_fn(args, kwargs, cfg, splits_target):
    bn, w, st = cfg
    num_tiles = triton.cdiv(WIDTH, bn)
    sp = min(splits_target, triton.cdiv(WIDTH, 64), num_tiles)

    def select(*a):
        return bn, w, num_tiles, sp

    old._select_config = select
    old._STAGES_OVERRIDE = st

    def fn():
        return old.qsa_sparse_paged_attention(*args, **kwargs)

    return fn


def main():
    results = []
    for shape in SHAPES:
        name = shape_name(shape)
        args, kwargs = make_inputs(shape)
        splits_target = h800_target(shape[0], shape[2], shape[6])

        ref_out = flaggems_vllm.qsa_sparse_paged_attention(*args, **kwargs)

        # sweep old op configs (reduced budget), keep best
        best = None
        for cfg in CONFIGS:
            fn = make_old_fn(args, kwargs, cfg, splits_target)
            try:
                fn()
                torch.cuda.synchronize()
            except Exception:
                continue
            try:
                lat = bench(fn, warmup=50, rep=100)
            except Exception:
                continue
            if best is None or lat < best[0]:
                best = (lat, cfg)
        assert best is not None, f"no valid config for {name}"
        best_lat, best_cfg = best

        best_fn = make_old_fn(args, kwargs, best_cfg, splits_target)
        abl_out = best_fn()
        max_diff = (abl_out.float() - ref_out.float()).abs().max().item()

        vllm_fn = lambda a=args, k=kwargs: vllm_qsa.qsa_sparse_paged_attention(*a, **k)
        prod_fn = lambda a=args, k=kwargs: flaggems_vllm.qsa_sparse_paged_attention(
            *a, **k
        )
        vllm_ms = bench(vllm_fn, warmup=1000, rep=100)
        abl_ms = bench(best_fn, warmup=1000, rep=100)
        prod_ms = bench(prod_fn, warmup=1000, rep=100)

        row = {
            "shape": name,
            "splits": splits_target,
            "best_cfg": best_cfg,
            "sweep_ms": best_lat,
            "vllm_ms": vllm_ms,
            "abl_ms": abl_ms,
            "prod_ms": prod_ms,
            "abl_speedup": vllm_ms / abl_ms,
            "prod_speedup": vllm_ms / prod_ms,
            "struct_marginal": abl_ms / prod_ms,
            "max_diff": max_diff,
        }
        results.append(row)
        print(
            f"{name:34s} sp={splits_target:2d} cfg={best_cfg} "
            f"vllm={vllm_ms:.4f} abl={abl_ms:.4f}({row['abl_speedup']:.3f}) "
            f"prod={prod_ms:.4f}({row['prod_speedup']:.3f}) "
            f"marg={row['struct_marginal']:.3f} diff={max_diff:.4f}",
            flush=True,
        )
        del args, kwargs
        torch.cuda.empty_cache()

    geo_abl = math.exp(sum(math.log(r["abl_speedup"]) for r in results) / len(results))
    geo_prod = math.exp(
        sum(math.log(r["prod_speedup"]) for r in results) / len(results)
    )
    geo_marg = math.exp(
        sum(math.log(r["struct_marginal"]) for r in results) / len(results)
    )
    summary = {
        "geo_abl": geo_abl,
        "geo_prod": geo_prod,
        "geo_struct_marginal": geo_marg,
    }
    print(
        f"\nGEOMEAN abl={geo_abl:.3f} prod={geo_prod:.3f} struct_marginal={geo_marg:.3f}"
    )
    out_path = "/tmp/qsa_ablation/ablation_result.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "rows": results}, f, indent=1)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
