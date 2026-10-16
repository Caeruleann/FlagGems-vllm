# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Small-T performance evidence collector for qsa_sparse_paged_attention.

Follows qsa_small_t_debug_guide.md: this round only collects evidence.
It reuses the checked-in benchmark input construction, times vLLM vs Gems
under three timing modes and two output modes, records launch metadata and
actual tuned configs, optionally captures CPU/GPU traces, and checkpoints
results.json + summary.csv as it goes.

This script is a debug tool, not production code; it may import torch
reference utilities for comparison exactly like the benchmark does.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import hashlib
import io
import json
import os
import pstats
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402
import triton  # noqa: E402

from benchmark.test_qsa_sparse_paged_attention import (  # noqa: E402
    QSASparsePagedAttentionBenchmark,
)
from flaggems_vllm.testing import assert_close  # noqa: E402

# T -> (num_query_heads, num_kv_heads, page_size, num_requests); see the guide.
SHAPES = {
    1: (24, 2, 1600, 2),
    16: (12, 1, 1600, 3),
    32: (6, 1, 800, 5),
    128: (24, 2, 1568, 7),
}


def command_output(command):
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<failed: {exc!r}>"


def file_identity(path):
    path = Path(path)
    try:
        digest = hashlib.md5(path.read_bytes()).hexdigest()[:12]
        return {
            "path": str(path),
            "size": path.stat().st_size,
            "mtime": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).isoformat(),
            "md5_12": digest,
        }
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "error": repr(exc)}


def tensor_metadata(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "strides": list(tensor.stride()),
        "numel": tensor.numel(),
        "device": str(tensor.device),
        "data_ptr_mod16": tensor.data_ptr() % 16,
    }


def jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return str(value)


def summarize(samples):
    rows = []
    grouped = {}
    for sample in samples:
        key = (sample["case"], sample["out_mode"], sample["mode"])
        grouped.setdefault(key, {}).setdefault(sample["provider"], []).append(
            sample["latency_us"]
        )
    for (case, out_mode, mode), providers in sorted(grouped.items()):
        row = {"case": case, "out_mode": out_mode, "mode": mode}
        for provider, values in providers.items():
            med = statistics.median(values)
            row[f"{provider}_median_us"] = med
            row[f"{provider}_min_us"] = min(values)
            row[f"{provider}_max_us"] = max(values)
            row[f"{provider}_spread_pct"] = (
                (max(values) - min(values)) / med * 100 if med else None
            )
        if "vllm" in providers and "gems" in providers:
            row["speedup"] = row["vllm_median_us"] / row["gems_median_us"]
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def operator_latency(fn, torch, rep_ms, warmup_ms):
    """Batch-of-calls wall timing with sync boundaries (benchmark口径)."""
    fn()
    torch.cuda.synchronize()
    probe_start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    per_call_s = max(time.perf_counter() - probe_start, 1e-6)
    n_warm = max(1, int(warmup_ms / 1000 / per_call_s))
    n_rep = max(1, int(rep_ms / 1000 / per_call_s))
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_rep * 1000


class observe_entries:
    """Probe kernel entry points to record launch metadata per submission.

    Wraps the ``run`` method of LibEntry (Gems) or JITFunction (vLLM)
    entries for the duration of the first call. Snapshots land in the
    ``snapshots`` dict under each entry name.
    """

    def __init__(self, entries, snapshots):
        self.entries = entries
        self.snapshots = snapshots
        self._restore = []

    def __enter__(self):
        for name, entry in self.entries.items():
            records = self.snapshots.setdefault(name, [])
            original = entry.run
            self._restore.append((entry, original))

            def wrapped(*args, _original=original, _records=records, **kwargs):
                record = {"args_tail": jsonable(list(args[-6:]))}
                grid = kwargs.get("grid")
                record["grid"] = None if callable(grid) else jsonable(list(grid))
                call_constexprs = {
                    key: value
                    for key, value in kwargs.items()
                    if isinstance(value, (int, float, bool, str))
                }
                result = _original(*args, **kwargs)
                # LibEntry.run returns (compiled_kernel, constexprs) where
                # constexprs includes the tuned META values chosen by the
                # libtuner (e.g. BLOCK_N / num_stages) plus the caller's
                # constexpr kwargs; JITFunction.run returns the compiled
                # kernel alone, so only call-site constexprs are known.
                tuned_constexprs = {}
                if isinstance(result, tuple) and len(result) == 2:
                    tuned_constexprs = dict(result[1] or {})
                record["constexprs"] = jsonable({**call_constexprs, **tuned_constexprs})
                compiled = result[0] if isinstance(result, tuple) else result
                for attr in ("n_regs", "n_spills"):
                    record[attr] = getattr(compiled, attr, None)
                metadata = getattr(compiled, "metadata", None)
                record["shared"] = getattr(metadata, "shared", None)
                record["num_warps_effective"] = getattr(metadata, "num_warps", None)
                _records.append(record)
                return result

            entry.run = wrapped
        return self.snapshots

    def __exit__(self, exc_type, exc, tb):
        for entry, original in self._restore:
            entry.run = original
        return False


def _kernel_rows_from_trace(trace_path):
    """Per-CUDA-kernel count / median / total from a chrome trace."""
    import collections

    payload = json.loads(Path(trace_path).read_text())
    events = payload.get("traceEvents", [])
    durations = collections.defaultdict(list)
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        if category not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        name = event.get("name", "<unnamed>")
        durations[name].append(event.get("dur", 0))
    rows = []
    for name, durs in sorted(durations.items()):
        rows.append(
            {
                "kernel": name,
                "count": len(durs),
                "median_us": statistics.median(durs),
                "total_us": sum(durs),
            }
        )
    return rows


def profile_callable(fn, entries, torch, out_base):
    """Capture torch profiler trace + CPU table + cProfile for one callable."""
    out_base = Path(out_base)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as profiler:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    trace_path = out_base.with_suffix(".trace.json")
    profiler.export_chrome_trace(str(trace_path))

    cpu_path = out_base.with_suffix(".cpu.txt")
    cpu_path.write_text(profiler.key_averages().table(sort_by="self_cpu_time_total"))

    kernel_rows = _kernel_rows_from_trace(trace_path)
    gpu_path = out_base.with_suffix(".gpu.csv")
    write_csv(gpu_path, kernel_rows)
    if not kernel_rows:
        out_base.with_suffix(".gpu_missing.txt").write_text(
            "no CUDA kernel events captured in the trace\n"
        )

    profiler_run = cProfile.Profile()
    profiler_run.enable()
    for _ in range(200):
        fn()
    profiler_run.disable()
    torch.cuda.synchronize()
    stats = pstats.Stats(profiler_run)
    python_path = out_base.with_suffix(".python.txt")
    buffer = io.StringIO()
    stats.stream = buffer
    stats.sort_stats("tottime").print_stats(40)
    buffer.write("\n\n===== cumulative =====\n")
    stats.sort_stats("cumulative").print_stats(40)
    python_path.write_text(buffer.getvalue())
    stats.dump_stats(str(out_base.with_suffix(".pstats")))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 32])
    parser.add_argument(
        "--kv-dtypes",
        nargs="+",
        choices=["bf16", "fp8"],
        default=["bf16", "fp8"],
    )
    parser.add_argument(
        "--out-modes",
        nargs="+",
        choices=["allocate", "preallocated"],
        default=["allocate", "preallocated"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["operator", "kernel", "cudagraph"],
        default=["operator", "kernel", "cudagraph"],
    )
    parser.add_argument("--rep-ms", type=float, default=20)
    parser.add_argument("--warmup-ms", type=float, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--index-kind", default="random")
    parser.add_argument("--count-mode", default="mixed")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args(argv)


def main(options=None):
    if options is None:
        options = parse_args()
    import importlib

    import flaggems_vllm

    gems = importlib.import_module("flaggems_vllm.ops.qsa_sparse_paged_attention")
    vllm = importlib.import_module("vllm.models.qwen4_exp.nvidia.ops.qsa")

    functions = {
        "vllm": vllm.qsa_sparse_paged_attention,
        "gems": flaggems_vllm.qsa_sparse_paged_attention,
    }
    entries = {
        "gems": {
            "main_entry": gems._QSA_MAIN_TUNED,
            "merge_entry": gems._QSA_MERGE_TUNED,
        },
        "vllm": {
            "main_entry": vllm._qsa_sparse_paged_gqa_splitk_kernel,
            "merge_entry": vllm._qsa_merge_splitk_kernel,
        },
    }
    options.out_dir.mkdir(parents=True, exist_ok=False)
    data = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "options": {**vars(options), "out_dir": str(options.out_dir.resolve())},
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "git_head": command_output(["git", "rev-parse", "HEAD"]),
            "git_status": command_output(["git", "status", "--short"]),
            "source": {
                "gems": file_identity(gems.__file__),
                "vllm": file_identity(vllm.__file__),
                "collector": file_identity(__file__),
                "libentry": file_identity(ROOT / "src/flaggems_vllm/utils/libentry.py"),
                "benchmark": file_identity(
                    ROOT / "benchmark/test_qsa_sparse_paged_attention.py"
                ),
                "tune_configs": file_identity(
                    ROOT / "src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml"
                ),
                "heuristics": file_identity(
                    ROOT
                    / "src/flaggems_vllm/runtime/backend/_nvidia/heuristics_config_utils.py"
                ),
            },
            "selected_env": {
                name: os.environ.get(name)
                for name in ("CUDA_VISIBLE_DEVICES", "DNN_VENDOR", "TRITON_CACHE_DIR")
            },
            "custom_tuning_database": bool(os.environ.get("FLAGGEMS_DB_URL")),
            "nvidia_smi_before": command_output(["nvidia-smi"]),
        },
        "cases": [],
        "samples": [],
    }

    def checkpoint():
        (options.out_dir / "results.json").write_text(
            json.dumps(data, indent=2, default=str) + "\n"
        )
        write_csv(options.out_dir / "summary.csv", summarize(data["samples"]))

    checkpoint()
    try:
        with torch.inference_mode():
            for num_rows in options.rows:
                for kv_dtype in options.kv_dtypes:
                    hq, hkv, page_size, requests = SHAPES[num_rows]
                    shape = [
                        num_rows,
                        hq,
                        hkv,
                        page_size,
                        True,
                        requests,
                        kv_dtype == "fp8",
                        options.index_kind,
                        options.count_mode,
                    ]
                    case_name = f"t{num_rows}_{kv_dtype}_{options.index_kind}_{options.count_mode}"
                    print(
                        f"Preparing {case_name}; first call may tune kernels...",
                        flush=True,
                    )
                    bench = QSASparsePagedAttentionBenchmark(functions["vllm"])
                    bench.shapes = [shape]
                    # Use the checked-in benchmark constructor unchanged.
                    packed = next(bench.get_input_iter(torch.bfloat16))
                    inputs, kwargs = packed[:-1], packed[-1]
                    case = {
                        "name": case_name,
                        "shape_spec": shape,
                        "input_metadata": [tensor_metadata(t) for t in inputs],
                        "gate": tensor_metadata(kwargs["output_gate"]),
                        "counts": inputs[3][:, -1].cpu().tolist(),
                        "launches": {},
                    }
                    data["cases"].append(case)
                    profile_calls = []
                    for out_mode in options.out_modes:
                        calls = {}
                        outputs = {}
                        for provider, op in functions.items():
                            call_kwargs = dict(kwargs)
                            if out_mode == "preallocated":
                                call_kwargs["out"] = torch.empty_like(inputs[0])
                            calls[provider] = (
                                lambda op=op, kw=call_kwargs, args=inputs: op(
                                    *args, **kw
                                )
                            )
                            snapshots = {}
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            with observe_entries(
                                entries[provider], snapshots=snapshots
                            ):
                                outputs[provider] = calls[provider]()
                            torch.cuda.synchronize()
                            case["launches"][f"{provider}_{out_mode}"] = {
                                "first_call_with_metadata_probe_ms": (
                                    time.perf_counter() - start
                                )
                                * 1000,
                                "entries": snapshots,
                            }
                        assert_close(
                            outputs["gems"],
                            outputs["vllm"],
                            dtype=inputs[0].dtype,
                            atol=2e-2,
                        )
                        del outputs
                        case.setdefault("correctness", {})[
                            out_mode
                        ] = "passed_vllm_comparison"
                        checkpoint()
                        for mode in options.modes:
                            for trial in range(options.repeats):
                                order = (
                                    ["vllm", "gems"]
                                    if trial % 2 == 0
                                    else ["gems", "vllm"]
                                )
                                for provider in order:
                                    fn = calls[provider]
                                    if mode == "operator":
                                        latency_ms = operator_latency(
                                            fn, torch, options.rep_ms, options.warmup_ms
                                        )
                                    elif mode == "kernel":
                                        latency_ms = triton.testing.do_bench(
                                            fn,
                                            warmup=options.warmup_ms,
                                            rep=options.rep_ms,
                                            return_mode="median",
                                        )
                                    else:
                                        for _ in range(3):
                                            fn()
                                        torch.cuda.synchronize()
                                        latency_ms = triton.testing.do_bench_cudagraph(
                                            fn, rep=options.rep_ms, return_mode="median"
                                        )
                                    data["samples"].append(
                                        {
                                            "case": case_name,
                                            "out_mode": out_mode,
                                            "mode": mode,
                                            "trial": trial,
                                            "order": order,
                                            "provider": provider,
                                            "latency_us": float(latency_ms) * 1000,
                                        }
                                    )
                            checkpoint()
                            print(f"  {out_mode}: {mode} collected", flush=True)
                        if options.profile:
                            for provider, fn in calls.items():
                                profile_calls.append((provider, out_mode, fn))
                    # Finish all timings for this case before either profiler.
                    for provider, out_mode, fn in profile_calls:
                        profile_callable(
                            fn,
                            entries[provider],
                            torch,
                            options.out_dir / f"{case_name}_{provider}_{out_mode}",
                        )
                    print(f"Finished {case_name}", flush=True)
        data["status"] = "complete"
    except Exception as exc:
        data["status"] = "failed"
        data["error"] = repr(exc)
        raise
    finally:
        data["environment"]["nvidia_smi_after"] = command_output(["nvidia-smi"])
        checkpoint()
    print(f"Results: {options.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
