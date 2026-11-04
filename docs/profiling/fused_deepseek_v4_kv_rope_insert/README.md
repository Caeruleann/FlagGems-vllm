# fused_deepseek_v4_kv_rope_insert NCU profile

## Environment

- Source commit: `e0da804ef15fb1f939b2da80c945d005f744a13c`
- Profile date: 2026-11-04
- GPU: NVIDIA H800, compute capability 9.0, 132 SMs
- Nsight Compute: 2026.3.1
- Torch: 2.11.0+cu130; FlagTree/Triton: 3.6.0
- Kernel: one token per program, one warp per program
- Cache block size: 32

The profiled source tree is identical to commit `eedb91d`; the commit was
later rewritten as `e0da804` without changing the tree.

## Data

Each `.log` file contains the raw NCU text output for one profiled launch,
with one unprofiled warmup launch before it. `summary.csv` contains the
corresponding extracted metrics. The `duration_us` values are measured under
NCU and must not be compared directly with benchmark latencies.

| Shape `(tokens, insert)` | Format | DRAM % | SM % | L1TEX stall % | Achieved occupancy % |
| --- | --- | ---: | ---: | ---: | ---: |
| `(20, 17)` | BF16 | 0.21 | 0.06 | 78.2 | 1.52 |
| `(20, 17)` | packed_v4 | 0.22 | 0.16 | 60.9 | 1.53 |
| `(8192, 8192)` | BF16 | 35.64 | 12.85 | 83.2 | 31.55 |
| `(8192, 8192)` | FP8 | 34.77 | 21.11 | 79.2 | 30.73 |
| `(8192, 8192)` | packed_v4 | 33.92 | 34.28 | 61.7 | 36.40 |
| `(8192, 8192)` | MXFP8 | 35.96 | 30.71 | 66.4 | 35.74 |

The 8K launches are memory-latency-bound rather than DRAM-bandwidth-bound.
They have 1.94 waves per SM, 50% theoretical occupancy, and no local or shared
memory spilling. The 20-token launches have only 17 CTAs on 132 SMs and are
dominated by insufficient grid parallelism and fixed latency.

## Reproduce

Run from the repository root. Set `CASE_TOKENS` to 20 or 8192 and
`CASE_FORMAT` to the desired cache format. The input helper uses 17 insertion
slots for the 20-token case and `num_tokens` insertion slots otherwise.

```bash
source /workspace/env.sh
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
CASE_TOKENS=8192 CASE_FORMAT=packed_v4 ncu \
  --target-processes all \
  --kernel-name "regex:fused_deepseek_v4_kv_rope_insert" \
  --launch-skip 1 --launch-count 1 \
  --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  python - <<'PY'
import os
import torch
import flaggems_vllm
from baseline.fused_deepseek_v4_kv_rope_insert import build_inputs

n = int(os.environ["CASE_TOKENS"])
insert = 17 if n == 20 else n
binding = {
    "num_tokens": n,
    "num_tokens_insert": insert,
    "cache_block_size": 32,
    "cache_format": os.environ["CASE_FORMAT"],
}
args, kwargs = build_inputs(binding, torch.bfloat16, flaggems_vllm.device)
flaggems_vllm.fused_deepseek_v4_kv_rope_insert(*args, **kwargs)
torch.cuda.synchronize()
flaggems_vllm.fused_deepseek_v4_kv_rope_insert(*args, **kwargs)
torch.cuda.synchronize()
PY
```

## Follow-up A/B data

See `ab_position_v4/README.md` for the paired CUDA/Triton measurements,
SourceCounters/SASS analysis, position-prefetch result, and packed V4 split
result.
