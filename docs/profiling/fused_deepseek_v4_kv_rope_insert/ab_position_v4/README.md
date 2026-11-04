# Position prefetch and packed V4 split A/B

## Protocol

- Base source: `e0da804ef15fb1f939b2da80c945d005f744a13c`.
- GPU: NVIDIA H800; NCU 2026.3.1; Torch 2.11.0+cu130;
  FlagTree/Triton 3.6.0.
- Shape: `num_tokens=num_tokens_insert=8192`, `cache_block_size=32`.
- Input seed: 20261104. CUDA and Triton receive the same tensors and the same
  `slot_mapping`; only the last slot is negative.
- Ordinary latency: nine repetitions per implementation, alternating their
  measurement order. Each repetition uses `triton.testing.do_bench` with
  25 ms warmup, 100 ms measurement, and median aggregation. The summary uses
  the median of the nine repetitions.
- NCU: one warmup launch and one profiled launch per implementation/format.
  SourceCounters and PC sampling were available. `sass_*.csv` preserves SASS
  order and per-instruction stall counts; `top_stalls.csv` lists the top 12
  stall locations per report.
- `duration_us` under NCU is not the ordinary benchmark latency. Keep/revert
  decisions use the ordinary paired measurements.

## Ordinary paired results

Times are milliseconds. SpeedUp is CUDA latency divided by Triton latency.

| Stage | Format | CUDA | Triton | SpeedUp |
| --- | --- | ---: | ---: | ---: |
| Before | packed_v4 | 0.015456 | 0.013216 | 1.1695 |
| Position prefetch | packed_v4 | 0.015360 | 0.013472 | 1.1401 |
| V4 split | packed_v4 | 0.015392 | 0.012544 | 1.2270 |
| Final | packed_v4 | 0.015328 | 0.012544 | 1.2219 |
| Before | mxfp8 | 0.015072 | 0.013088 | 1.1516 |
| Position prefetch | mxfp8 | 0.015232 | 0.013120 | 1.1610 |
| Final | mxfp8 | 0.015200 | 0.013088 | 1.1614 |

All nine individual samples per stage/format are in `paired_samples.csv`.

## Experiment 1: position prefetch

The original SASS issues the `slot_mapping` 64-bit load, evaluates the negative
slot predicate, and exits before issuing the `position_ids` 64-bit load. A
high-stall LEA consumes `position_ids`, and a high-stall RoPE FMUL consumes
cos/sin data whose address depends on it.

Moving the source-level position load above the negative-slot check did not
move the SASS load above the EXIT; the compiler kept it on the live-token
path. Packed V4 latency increased from 0.013216 to 0.013472 ms, while MXFP8
changed only from 0.013088 to 0.013120 ms. The change was reverted.

## Experiment 2: packed V4 split

The final source keeps a 512-wide tile but masks the last 64 values for the
NoPE load. The first seven groups are quantized independently, while the last
64 values are loaded, rotated, rounded to BF16, and stored directly. It no
longer loads the RoPE tail twice, reassembles a full rotated row, or quantizes
that row before writing V4.

| SASS / NCU metric | Before | V4 split |
| --- | ---: | ---: |
| SASS instructions | 336 | 224 |
| LDG instructions | 10 | 5 |
| SHFL instructions | 11 | 3 |
| FFMA instructions | 27 | 13 |
| Registers per thread | 55 | 37 |
| NCU duration (us) | 9.09 | 9.25 |
| DRAM throughput (%) | 35.05 | 34.40 |
| Achieved occupancy (%) | 36.89 | 30.27 |

The split reduces ordinary packed V4 latency by 5.1% despite no improvement
in NCU duration or occupancy. The keep decision is therefore based on the
paired non-profiling latency and the reduced SASS work, not on a claim that
all NCU metrics improved. MXFP8 remains on the original path and its final
latency is unchanged.

## Validation

- 61 functional tests passed, including whole-cache byte equality.
- Ordinary core benchmark: 4 passed.
- CUDA Graph core benchmark: 4 passed.
- CUDA Graph comprehensive benchmark: 4 passed.

`paired_summary.csv` and `sass_summary.csv` contain the machine-readable
summaries. `ncu_*.log` contains the raw NCU details output.
