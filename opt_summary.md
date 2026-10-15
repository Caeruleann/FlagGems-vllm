# QSA Sparse Attention H800 优化总结

对应计划:[opt.md](opt.md)(H800 上 `qsa_sparse_paged_attention` 的 BF16 decode/prefill 优化)。
执行日期:2026-10-14。验收设备:NVIDIA H800(sm_90,单卡),torch 2.11.0+cu130,triton 3.6.0,vLLM 0.31.0。

## 1. 执行的事

按 opt.md 的「配置 → 空 split → 主循环 → merge」顺序逐轮试验,每轮只做一类变化,validate → benchmark → keep/revert。Trial 明细见第 4 节。

### 测试口径先行(计划第 2 节)

- 保留原 14 组 shape 作为回归集合;
- `tests/test_qsa_sparse_paged_attention.py` 新增 97 项用例:
  - 索引分布:随机稀疏 + **连续 4 token 成组**(模拟 QSA 局部性,直接构造,不恢复 indexer);
  - 行长度:L ∈ {0, 1, 17, 31, 32, 33, 63, 64, 65, 2048, 2051} + 长短混合(半行 17 / 半行全宽);
  - K/V layout:原非连续 split view + 新增连续 K/V 对照;
  - CUDA Graph 捕获→换内容重放 freshness(验证不读旧 workspace);
- 精度不变:仍用框架 `assert_close` 同时对照冻结 vLLM op 与 Torch reference,阈值未放宽。

### R1:H800 配置与 autotune(keep)

- 在 H800 上扫描 `BLOCK_N {32,64,128} × warps {1,2,4,8} × stages {1,2,3} × splits {1..64}`(先粗扫 split,再细调 block/warp/stage,非全笛卡尔积)。
- **split 数由 host 固定**(计划要求),H800 实测表(bp = T×Hkv):bp≤32→64;32<bp≤256→2;256<bp≤512→4;>512→1;FP8 保持 legacy 表;非 H800 的 sm90(H20)/sm120 保留原表。
- 主 kernel 接入 `runtime.get_tuned_config("qsa_sparse_paged_attention")` + `@libentry` + `@libtuner`:25 个候选(yaml 显式枚举,按扫描证据剪掉超资源的组合),key = num_rows(log)/Hq/GQA/D/选择宽度/page size/split 数/关键 stride,KV dtype 随张量自动入 key;merge kernel 独立 autotune(num_warps),key 含 NUM_SPLITS/HEAD_DIM/TOPK/BLOCK_N。
- **host fast-path**:纯 libtuner 接入使 decode 回退(LibEntry 命中路径 45.4µs/launch,plain jit 32.6µs;decode 为 host-bound)。改为以完整元数据(形状/stride/dtype/对齐/scale)为键缓存启动计划,首个签名走 LibEntry 完成 tuner 选型,其后直接 `CompiledKernel[grid]` 启动(runner ~8µs/launch)。全量校验仍在每个新签名上执行,校验只依赖键内字段,语义不变。decode 每次调用 host 开销 106µs → 33µs(vLLM op 93µs)。
- `tune_configs.yaml` 新增 `qsa_sparse_paged_attention`(25 项)与 `qsa_sparse_paged_attention_merge`(3 项);无 yaml 的厂商回退默认配置。

### R2:减少空 split 和短行开销(keep)

- 主 kernel:`split_id > 0` 且该行有效 tile 数 ≤ split_id 时,**在加载 Q 之前退出**;
- 仅需一个 tile(含 L=0)的行由 split 0 **直接写最终输出**(先 BF16 舍入再 FP32 gate,与 merge 数学等价),merge 跳过该行;
- merge 用同一有效 split 数 `min(ceil(L/BLOCK_N), NUM_SPLITS)` 屏蔽 partial 读取 —— 空 split 不再写 workspace,merge 也绝不会读到未初始化/陈旧数据(CUDA Graph 换内容重放安全);
- 所有判断由 kernel 读 count 后在 GPU 完成,无 CPU 同步。

### R3:主循环访存与流水线(停止,无收益)

- ncu 取证(T=700,b128w4s2):Duration 359µs,**DRAM 74%(2.49TB/s)**,SM 20%,240 寄存器/线程,占用率 6.25%;stall 以 wait(1.48/issue)+ long_scoreboard(1.13)为主。
- E2「V 加载移到 QK dot 之后」:各 shape ratio 0.97–1.015,噪声内 → revert。
- E3「索引/页表手动预取一级流水」:T=1024 0.81–0.91、T=128 0.65,显著变差(寄存器压力 + 索引运算重复,Triton 内置 pipeliner 已更优)→ revert。
- 占用率修复尝试(b128+8 warps 降每线程寄存器)反而更差(0.44 vs 0.34 ms)——MMA 效率主导。
- 两次平/负试验 + 瓶颈定位,按 opt.md 停止规则终止主循环调优。

### R4:merge 优化(负收益,保留现状)

- 对照「一个 Q head 一个 program」vs「一个 KV head 合并多个 Q head」:grouped 变体 T=1 0.28× / T=16 0.35× / T=32 0.28× / T=128 0.62× —— merge 是 latency-bound 小 kernel,细粒度分解并行度更优 → revert。
- merge 的 num_warps 已按计划在 R1 接入独立 autotune。

## 2. 优化效果与性能前后对比

口径:`pytest -s benchmark/test_qsa_sparse_paged_attention.py --level core --iter 20 --warmup 5`,
SpeedUp = vLLM 耗时 / FlagGems 耗时(整 op,含主 kernel + merge + host)。
优化前为冻结的原始 port(与 vLLM op 逐字节一致);优化后为两轮独立复测的平均。

### BF16(主要优化对象)

| Shape | 优化前 vLLM (ms) | 优化前 Gems (ms) | 优化前 SpeedUp | 优化后 vLLM (ms) | 优化后 Gems (ms) | **优化后 SpeedUp** |
| --- | --- | --- | --- | --- | --- | --- |
| T=1 (decode) | 0.0356 | 0.0357 | 0.996 | 0.0327 | 0.0156 | **2.10** |
| T=16 | 0.0378 | 0.0398 | 0.951 | 0.0354 | 0.0261 | **1.36** |
| T=32 | 0.0387 | 0.0393 | 0.985 | 0.0371 | 0.0380 | **0.98** |
| T=128 | 0.2093 | 0.2093 | 1.000 | 0.2094 | 0.1485 | **1.41** |
| T=257 | 0.1832 | 0.1837 | 0.997 | 0.1810 | 0.1492 | **1.21** |
| T=513 | 0.3322 | 0.3313 | 1.003 | 0.3305 | 0.2596 | **1.27** |
| T=700 | 0.4508 | 0.4515 | 0.998 | 0.4502 | 0.3380 | **1.33** |
| T=1024 | 1.3040 | 1.3053 | 0.999 | 1.2939 | 0.9825 | **1.32** |
| T=2048 (prefill) | 2.6170 | 2.6224 | 0.998 | 2.5911 | 1.8918 | **1.37** |
| T=2048 (decode) | 1.9038 | 1.9041 | 1.000 | 1.8981 | 1.8955 | **1.00** |

### FP8(保持正确性与现有性能,实际同步提升)

| Shape | 优化前 SpeedUp | **优化后 SpeedUp** |
| --- | --- | --- |
| T=1 | 1.027 | **2.07** |
| T=128 | 1.002 | **1.51** |
| T=2048 (prefill) | 1.000 | **1.68** |
| T=2048 (decode) | 0.999 | **1.61** |

### 汇总

- 14 组 shape 全部 ≥ 0.9(最低 T=32 0.98),无回归;
- BF16 **decode 几何平均 ≈ 1.41**,**prefill 几何平均 ≈ 1.32**,均 ≥ 1.1;
- FP8 几何平均 ≈ 1.70(非首轮目标,仅随 host/kernel 改进被动提升);
- 短行场景(R2 动机,L=17/长短混合,harness 分段计时):0.97–1.08×(raw 启动口径),长行无回退;
- decode 每次调用 host 开销 93µs(vLLM)→ 33µs;T=1 整 op GPU 真实耗时(main 4.9µs + merge 2.6µs,nsys) unchanged in shape 但空 split 消除后有效吞吐提升。

## 3. 正确性与工程结论

- `pytest -q tests/test_qsa_sparse_paged_attention.py`:111 项全过(14 回归 + 96 新口径 + 1 graph freshness);quick 模式 101 项;
- 生产路径无 torch compute fallback 审计通过(仅元信息读取 + `torch.empty*` 分配);
- black / isort / flake8 通过;
- 编译/autotune 与运行时间分离:全 bucket 首轮 tuning 一次性约 2–4 分钟,结果持久化于 LibCache sqlite(`~/.flagdnn/config_cache`),之后进程/CI 直接命中,预热后即为上表读数;
- 不修改 public API、分页索引、GQA、gate、数值精度;FP32 accumulator、在线 softmax、64 位缓存地址、FP8 转换路径均未改动。

## 4. Trial 记录(keep/revert)

| Trial | 假设 | 结果 | 决定 |
| --- | --- | --- | --- |
| R1a | libtuner 接入 + host 固定 split | decode 0.74–0.81(host-bound 暴露) | 需 R1b |
| R1b | fast-path 计划缓存,直接启动编译产物 | decode 1.95–2.07,全部 ≥0.94 | keep |
| R2 | 空 split 早退 + merge 屏蔽 + 单 tile 直写 | decode 2.08/1.35/0.98,长行无回退 | keep |
| R3-E2 | V 加载后移 | 0.97–1.015,噪声内 | revert |
| R3-E3 | 索引/页表手动预取 | 0.65–0.99,显著变差 | revert |
| R4 | merge 按 KV head 合并 Q heads | 0.28–0.62,显著变差 | revert(保留现 merge) |

瓶颈结论(后续如需 deep opt):主 kernel 在 prefill 区为 DRAM 受限(74%,随机 gather 无 L2 复用,流量已不可约),低占用率由 MMA 片元寄存器决定,加 warps 反噬;跨 query 复用与索引排序按计划暂缓(需测量证据)。

## 5. 复现

```bash
# 功能(快速)
pytest -q tests/test_qsa_sparse_paged_attention.py --quick
# 功能(全量,首轮含 tuning)
pytest -q tests/test_qsa_sparse_paged_attention.py
# 性能
pytest -q -s benchmark/test_qsa_sparse_paged_attention.py --level core --iter 20 --warmup 5
```
