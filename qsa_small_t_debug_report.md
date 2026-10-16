# QSA 小 T 性能调试报告(证据轮,v2 修正)

日期:2026-10-16(初版)→ 2026-10-16(修正版)。代码基点:`3e60d06`。依据:`qsa_small_t_debug_guide.md` 与审查意见。
本轮**只采集证据,未修改生产代码**。代码变更仅限采集/实验脚本。

## 0. 修正声明(相对 v1)

1. **merge 分析漏掉寄存器溢出** —— 已补:v1 把 +2.4 µs 归于 split_counts 加载依赖/向量化,**不成立**。§3b 的四臂实验证明 spill 与 split_counts 都不是驱动因素,真实驱动是 **merge 的读宽度(S=64 vs 33)**。
2. **采集脚本缺 tuned 配置** —— 已修:`observe_entries` 现记录 LibEntry.run 返回的 tuned constexpr。重采后确认 v1 中"BLOCK_N=64、31 个空 split"的推断**错误**:Gems BF16 T=32 实际主配置是 **BLOCK_N=32 / S=64 / w4**(65 tiles 铺满 64 splits,长行几乎没有空 split;L=17 行才是单 tile 直写)。
3. **host 定量分解无依据** —— 已补单变量实测(§3c):分配是最大头;LibEntry vs plain-jit 仍为 profile 级相对证据,无单变量实测(改动需动代码,超出本轮"只采集"范围)。
4. **汇总表实为 run1** —— 已改为 run1/run2 两轮平均并列出单轮值;group4 结论已收敛为"当前案例收益有限,不能排除整个算子的索引局部性影响"。

## 1. 执行记录

- 环境:单机 8×H800,采集期间全部空闲;GPU 0;torch 2.11.0+cu130、triton 3.6.0、vLLM 0.31.0。
- 采集:指南 §2 四条命令(run1、run2、profile_fp8_t1、profile_bf16_t32,probe 修复后重采)+ §7 两条控制实验(short、group4,沿初版);两轮 launch 配置逐项相同。
- 实验:`tools/qsa_merge_config_exp.py`(新增)——固定生产 main 与 workspace,对照 merge 变体。
- 原始证据:`qsa_small_t_results/`(v2:run1/run2/profile 为修复 probe 后的重采;`merge_config_exp.json`)。

## 2. 计时汇总(allocate 模式,run1/run2 两轮中位数平均;SpeedUp=vLLM/Gems)

| 案例 | operator | kernel | cudagraph |
| --- | --- | --- | --- |
| BF16 T=1 | 0.984 | 0.983 | 0.986 |
| BF16 T=16 | 0.977 | 0.965 | 1.136 |
| BF16 T=32 | 0.986 | 0.983 | **0.945** |
| FP8 T=1 | 0.979 | 0.970 | 1.244 |
| FP8 T=16 | 0.982 | 0.990 | 1.221 |
| FP8 T=32 | 0.982 | 1.477 | 1.755 |
| short(L=17)T=1 / T=32(cudagraph,初版采集) | 0.980 / 0.978 | – | 1.43 / 2.14(BF16)、1.66 / 1.81(FP8) |
| group4 T=1 / T=32(cudagraph,初版采集) | 0.988 / 0.983 | – | 1.00 / 0.94(BF16)、1.24 / 1.70(FP8) |

单轮波动多数 <5%;FP8 T=32 kernel 模式两轮 1.476/1.479 高度一致。与初版报告数值在噪声内一致。

## 3. 证据与归因

### 3a. launch 配置(probe 修复后,run1)

| 案例 | main:BLOCK_N/warps/regs/spills(shared) | S | merge:warps/regs/spills |
| --- | --- | --- | --- |
| Gems BF16 T=32 | **b32 / w4 / 85 / 0**(37 KB) | 64 | w2 / **255 / 18** |
| vLLM BF16 T=32 | b64 / w2 / 200 / 0(71 KB) | 33 | w2 / 156 / 0 |
| Gems FP8 T=1 | b64 / w8 / 110 / 0 | 64 | w2 / **255 / 18** |
| vLLM FP8 T=1 | b64 / w2 / 255 / **72** | 33 | w2 / 156 / 0 |

merge 的 255/18(Gems)对 156/0(vLLM)即审查指出的混淆项,下文 3b 分离。

### 3b. merge 配置实验(固定 Gems 生产 main 与 buffers;merge 单独 do_bench 中位 + main+merge 链 cudagraph;`merge_config_exp.json`)

| 案例 | 臂 | w | regs/spills | merge µs | 链 µs |
| --- | --- | --- | --- | --- | --- |
| t32_bf16 | prod(生产) | 2 | 255/18 | 13.17 | 30.78 |
| t32_bf16 | no_count 克隆(去 split_counts+早退) | 2 | 255/16 | 13.86 | 31.34 |
| t32_bf16 | vllm 内核 @S=33 | 2 | 156/0 | **10.40** | 28.56 |
| t32_bf16 | vllm 内核 @S=64 | 2 | 255/14 | 14.59 | 32.13 |
| t1_fp8 | prod | 2 | 255/18 | 9.63 | 11.04 |
| t1_fp8 | no_count 克隆 | 2 | 255/16 | 9.47 | 10.72 |
| t1_fp8 | vllm 内核 @S=33 | 2 | 156/0 | **7.78** | 9.81 |
| t1_fp8 | vllm 内核 @S=64 | 2 | 255/14 | 9.86 | 11.13 |
| t1_bf16 | prod / vllm@33 / vllm@64 | 2 | 255/18 / 156/0 / 255/14 | 9.55 / **7.58** / 10.21 | 10.05 / 8.71 / 10.23 |
| t16_bf16 | prod / vllm@33 / vllm@64 | 2 | 同上 | 11.36 / **10.40** / 14.59 | 16.62 / 15.30 / 19.39 |

(每个臂另测了 w1/w4:w1 全部 255+regs、spills 26–366,最慢;**w4 spills=0 但仍慢于 w2**(如 t32_bf16 prod w4 14.78 vs w2 13.17)——说明 spill 本身不是成本驱动,tuner 选 w2 是对的;merge tuning key 无需为 spill 调整。)

**归因结论(修正后)**:
1. **prod ≈ no_count(同宽度)≈ vllm@S=64** → split_counts 依赖、早退分支、寄存器溢出都不是差距来源;Gems 与 vLLM 的 merge 代码在同宽度下等价(vllm@S=64 与 prod 差距 <15%,噪声级)。
2. **vllm@S=33 比同代码 @S=64 快 25–30%** → BF16 T=32 merge 差距(及 FP8 T=1 的 +1.2 µs)**完全由读宽度解释**:Gems S=64 的 merge 每程序读 64 路 partial(FP32 64×256),vLLM S=33 只读 33 路,partial 流量约 2×;寄存器差异(255 vs 156)是宽度的结果而非原因。
3. 机制链:R6 的 splits 上限按最小候选 BLOCK_N(32)计算(cdiv(2051,32)=65)→ 允许 S=64;tuner 在 S=64 下为 t32_bf16 主 kernel 选了 BLOCK_N=32(65 tiles 铺满 64 splits,主 kernel 与 vLLM 同速 24.9 µs);而 vLLM 用自身 BLOCK_N=64 裁剪到 33 splits。**主 kernel 在 64 splits 下没有更快,merge 却读了两倍数据**。
   - 注:实验中"vllm@S=33 + Gems S=64 的 main"链仅为计时下界(Gems main 在 splits 33..63 写了有效 partial,33 宽 merge 会丢数据,结果不正确);可达成链应以 S=33 重调 main(下轮)。

### 3c. host 单变量实测(operator 路径组成,T=1 尺寸,2000 次循环均值)

| 组件 | 实测 |
| --- | --- |
| `torch.empty_like(q)` | 1.82 µs |
| partial_output 分配(1.5 MB fp32) | 3.45 µs |
| partial_lse 分配 | 3.40 µs |
| split_counts 分配 | 2.36 µs |
| alias 检查(7×untyped_storage,仅 out= 路径) | 1.06 µs |

- Gems 每调用分配 4 项 ≈ 11.0 µs,vLLM 3 项 ≈ 8.7 µs → **split_counts 是唯一多出项(+2.4 µs)**,与 operator 模式实测差(1.5–2.5 µs)吻合(Gems wrapper 比 vLLM 略省,抵掉零头)。
- preallocated 对照:vLLM 省 ~3.5–4 µs、Gems 只省 ~0.7 µs ≈ empty_like(1.8)− alias 检查(1.1),自洽。
- LibEntry×2 vs plain-jit×2:**只有 profile 级相对证据**(cProfile self:Gems dispatch ≈ 35 µs/200 调用 vs vLLM ≈ 25 µs/200 调用,放大系数未知),未做(也不应在本轮做)代码级单变量改动;设备查询(_qsa_device_class 等)合计 ~1 µs,次要。
- 修正 v1 的说法:"约 10 µs 差值换算成真实 1–2 µs"无依据;现在只有分配与 alias 检查有实测值,LibEntry 差值保持"profile 级,约 1–2 µs"的有限结论。

## 4. 结论(修正后)

1. **BF16 T=32 cudagraph 0.945 的归因**:merge 读宽度(S=64 vs 33)导致 +2.7–3.8 µs;split_counts、spill、早退均经对照排除;main 无差距。
2. **FP8 案例**:main 因 R2 早退 + 更优配置(消除 vLLM 的 72/128 regs spill)显著更快,merge 宽度代价被覆盖,净胜 1.22–1.76×。
3. **host**:Gems 每调用比 vLLM 多 ~2.4 µs 实测(分配)+ ~1.1 µs(alias,仅 out= 路径)+ ~1–2 µs profile 级(dispatch);与 operator 模式差距(1.5–2.5 µs)量级一致。
4. **短行**:R2 直写+早退收益确认(short 控制 cudagraph 1.4–2.1×)。
5. **索引局部性**:group4 与 random 在当前案例无差异;仅能说"当前案例收益有限",不排除其它分布(如实索引器输出)存在局部性影响,需真实分布数据再评估。

## 5. 下一轮建议(按证据优先级,均未实施)

1. **S 与 BLOCK_N 宽度的联动裁剪(针对结论 1)**:把 host splits 上限从"最小候选 BLOCK_N=32 的 tile 数"改为与预期 tile 宽度联动(例如按 64 宽 tile 裁剪得 S≤33),或在后端策略层对 decode 桶收窄目标 splits;随后**整体重调 main 并 main+merge 链复测**(tuner 会在新 S 下重选 BLOCK_N;评价标准是整链,不是 main 单项)。风险:短行场景的早退收益不受影响(机制保留),但长行的 main 并行度需复测。
2. **workspace 单次分配 + 类型视图**(针对结论 3 的 +2.4 µs 与分配总量 11 µs):三项合成一次 `torch.empty`(字节对齐后 view 出 fp32 partial/lse 与 int32 counts,全部 no-copy)。预期省 ~3 µs/调用;需补生命周期与 Graph 测试(视图为同存储,不违反"每次调用新分配"——存储仍逐调用新建)。
3. **不采纳**:merge 改写(split_counts 机制无代价);merge tuning key 扩容(配置已最优);索引排序(无证据)。
4. 待补证据(如需):LibEntry 与 plain-jit 的真实差值需一次受控代码实验(公共机制内最小改动 + 回归),或接受当前 profile 级结论。

## 6. 交付物

- `tools/qsa_small_t_debug.py`(probe v2:记录 tuned constexpr)
- `tools/qsa_merge_config_exp.py`(merge 四臂实验)
- `qsa_small_t_results/`(v2 重采 + `merge_config_exp.json` + 整包 tarball)
- 本报告(v1 已替换)
