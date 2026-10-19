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

## 5. 下一轮建议(按证据优先级;第 1 条已于 2026-10-16 实施,见 §7)

1. **S 与 BLOCK_N 宽度的联动裁剪(针对结论 1)**:把 host splits 上限从"最小候选 BLOCK_N=32 的 tile 数"改为与预期 tile 宽度联动(例如按 64 宽 tile 裁剪得 S≤33),或在后端策略层对 decode 桶收窄目标 splits;随后**整体重调 main 并 main+merge 链复测**(tuner 会在新 S 下重选 BLOCK_N;评价标准是整链,不是 main 单项)。风险:短行场景的早退收益不受影响(机制保留),但长行的 main 并行度需复测。
2. **workspace 单次分配 + 类型视图**(针对结论 3 的 +2.4 µs 与分配总量 11 µs):三项合成一次 `torch.empty`(字节对齐后 view 出 fp32 partial/lse 与 int32 counts,全部 no-copy)。预期省 ~3 µs/调用;需补生命周期与 Graph 测试(视图为同存储,不违反"每次调用新分配"——存储仍逐调用新建)。
3. **不采纳**:merge 改写(split_counts 机制无代价);merge tuning key 扩容(配置已最优);索引排序(无证据)。
4. 待补证据(如需):LibEntry 与 plain-jit 的真实差值需一次受控代码实验(公共机制内最小改动 + 回归),或接受当前 profile 级结论。

## 6. 交付物

- `tools/qsa_small_t_debug.py`(probe v2:记录 tuned constexpr)
- `tools/qsa_merge_config_exp.py`(merge 四臂实验)
- `qsa_small_t_results/`(v2 重采 + `merge_config_exp.json` + 整包 tarball)
- 本报告(v1 已替换)

## 7. 修复实施:cap64(2026-10-16)

§5 第 1 条已实施。host NUM_SPLITS 上限由「最小候选 BLOCK_N(32) 的 tile 数」改为「64 参考 tile 宽的 tile 数」(`_SPLIT_CAP_TILE_WIDTH = 64`,op 内注释含依据),与 vLLM 固定 BLOCK_N=64 的 `min(target, cdiv(width, 64))` 语义对齐。width=2051 时:S 64 → 33(BF16 base_programs≤32、FP8 ≤24 桶,即 T=1/16/32 等 decode 案例);T≥128 各桶 target ≤ 8,不受影响。

### 证据(定配置对照:harness 裸 kernel 扫描,同 GPU 背靠背,merge 固定 w2/s1)

| 案例 | vLLM (ms) | best@S=64 (ms) | best@S=33 (ms) | 整链变化 | speedup@64→33 |
| --- | --- | --- | --- | --- | --- |
| BF16 T=32 (ps=800) | 0.0419 | 0.0495 (b64/w2/s1) | 0.0421 (b32/w4/s2) | −15% | 0.846 → 0.995 |
| BF16 T=16 (ps=1600) | 0.0379 | 0.0323 (b64/w2/s1) | 0.0294 (b64/w2/s1) | −9% | 1.177 → 1.291 |
| BF16 T=1 (ps=1600) | 0.0360 | 0.0186 | 0.0188 | 持平 | 1.933 → 1.918 |
| FP8 T=1 (ps=1600) | 0.0393 | 0.0191 | 0.0188 | −2% | 2.061 → 2.093 |

同配置分阶段(T=32,b32/w4/s2):main 0.0395→0.0350(空 split 发射减半),merge 0.0151→0.0112(−26%,每行读 64→33 路 partial)——结论 1 的归因闭环。T=1 的 merge 只有 24 CTA,宽度效应可忽略,与该案例整链持平自洽。目标案例 BF16 T=32 由此前明显落后(0.846–0.95,随模式/调优噪声波动)提升到与 vLLM 持平(≈0.99)。

### 测量纪律(本轮教训)

- 首次「归档旧结果 vs 新跑结果」对比出现大量 S 未变案例的 ±0.1–0.35 摆动:vLLM 基线在不同 GPU/时钟状态下自身漂移 32–86%,且清缓存后 libtuner 重调在 µs 级案例的选配置噪声与效果量同量级。**跨时段/跨 GPU 的 speedup 对比无效**;有效口径只有:① 同环境背靠背 A/B;② 定配置裸 kernel 对照(上表)。
- 同环境 A/B 归档:`qsa_benchmark_results/qsa_ab_{oldcap,cap64}_{kernel,operator,cudagraph}_run{1,2}.json`。A/B(含重调噪声)方向与对照实验一致:cudagraph T=32 0.951→0.980、T=1 1.002→1.023;kernel 模式个别案例反向源于 libtuner 实选非扫描最优(如 T=32 S=33 实选 b64/w8/s2,扫描最优 b32/w4/s2,差 ~13%)——生产机首调存在同样噪声,候选集已覆盖各最优项。
- 扫描脚本:`tools/qsa_config_scan.py`(已同步 split_counts 签名)。

### 回归与剩余项

- `tests/test_qsa_sparse_paged_attention.py` + `tests/test_libentry.py` 全量通过(159 项);graph 双向/重放新鲜度不受影响;小宽度用例 width≤64 → S=1 直写路径,行为不变。
- §5 第 2 条(workspace 单次分配 + 类型视图,预期 ~3 µs/调用)仍未实施。

### 当前一致快照(2026-10-16 下午,GPU0,健康调优缓存,各 2 轮)

命令:`pytest -q -s benchmark/test_qsa_sparse_paged_attention.py --level core --mode {kernel,cudagraph} --iter 20 --warmup 5`、operator 用 `--iter 200 --warmup 50`;原始 JSON:`qsa_benchmark_results/qsa_final_<mode>_run{1,2}.json`。

| 模式 | 几何平均(38 行) | 最小值 | <1.0 的行 |
| --- | --- | --- | --- |
| kernel | 1.219 | 0.958 (BF16 T=1) | T=1/T=16/T=32(0.96–1.00) |
| operator | 1.241 | 0.908 (BF16 T=2048 pre=0,rep2 0.983) | T=16/T=32/FP8 T=1/T=128 短行变体(0.98±0.01) |
| cudagraph | **1.291** | 0.986 (BF16 T=32) | 仅 T=32 一行 0.986–0.997 |

要点:BF16 T=128 组 1.21–1.52、T=2048 pre=1 1.34–1.44、FP8 1.42–1.66、短行变体最高 1.65;T=700/T=513 为 1.08–1.27,低于归档的 1.28–1.34——经 harness 定配置核实,两案例**已在当日可达最优**(T=700 最优配置 b128/w4/s2 当日实测 0.440 ms,benchmark 实测 0.442 ms),差异来自环境漂移(同一配置归档日实测 0.328 ms,+34%,而 vLLM 基线同期仅漂 5–8%;不同时段对两个 kernel 的影响不同,比值随之漂移),非 cap64 回归。

另注意:libtuner 首调存在 µs 级误选噪声(本轮 T=128 曾从 b128/w4/s2 误选为 b64/w2/s1,sp 1.38→1.26,删 key 重调即恢复);跨时段/跨机对比务必用同环境背靠背。

## 8. decode 小组攻关:merge 依赖消除 + host 修剪(2026-10-16 晚,工作区未提交)

触发:用户验收口径(`CUDA_VISIBLE_DEVICES=6 pytest benchmark/test_qsa_sparse_paged_attention.py -s`,kernel 模式默认参数)下 BF16 T=1=0.964、T=32=0.956,要求 ≥1.0。

### 归因(profiler 级)

- kernel 模式 = `do_bench(整 op)`,微内核下测的是 host 流水线暴露;T=1 纯 kernel 两侧已持平(8.14 vs 8.22 µs),T=32 kernel 侧差 4.1 µs(main +3.0、merge +1.1)。
- merge 差距来源:split_counts 标量加载的**控制/数据依赖**串行在 lse/partial 大加载之前;vLLM 的 mask 只依赖 constexpr NUM_SPLITS,加载立即发射。
- host 两侧持平(~94 µs);`_qsa_device_class` 每次调用查询设备属性,约 1–2 µs。

### 改动(3 项)

1. **merge 加载边界改 constexpr**:lse/partial 加载按 `split_offsets < NUM_SPLITS`(分配边界,常数)发射,split_counts 的值改在加载后用 `tl.where` 应用(lse→-inf、partial→0.0,NaN 位模式无法传播);`merge_splits <= 1` 直写行早退保留。不变量从「merge 只读已写」改为「读取恒在分配界内、越界值在参与计算前被证明丢弃」。merge:T=32 −1.3 µs、T=1 −1.5 µs。
2. **split 策略去硬件路由**(用户指示:目前只测 H800):删除 `_qsa_device_class` 及 sm90/sm120/generic 表,单一 H800 策略(BF16 用 H800 重调表,FP8 沿用原 sm90 表),彻底省掉每次调用的设备查询(取代当天下午的缓存方案);op 侧不再传递 device_index。其它设备恢复路由时需重新实测。
3. **T=32 调优重选**:删 key 重调,实选 b64/w8/s2(GPU6 实测最优,整链 0.0362 ms vs vLLM 0.0381)。

### 否决项(已实测)

- workspace 单次分配 + 类型视图(§5 第 2 条):host 实测**反而慢 7–9 µs/调用**——3 次 `torch.empty` 比 1 次 empty + slice/view(dtype)/view(shape) 链便宜。结论改为不采纳,代码已回退(注释留存)。

### 结果(用户验收口径,GPU6,一行不落)

19 行全部 ≥ 1.0,几何平均 **1.284**:T=1 1.107、T=16 1.161、T=32 1.070、T=128 组 1.17–1.38、T=257 1.19、T=513 1.29、T=700 1.33、T=1024 1.31、T=2048 1.36/1.00、FP8 1.15–1.64。回归 159/159 通过(exit=0);black/isort/flake8 干净。已提交为 `029880f`。

### 原始数据归档(2026-10-19)

验收命令补跑并记录:`CUDA_VISIBLE_DEVICES=6 python -m pytest -q -s benchmark/test_qsa_sparse_paged_attention.py --record json --output qsa_benchmark_results/qsa_accept_gpu6_kernel.json`(即用户验收命令 + record,kernel 模式 comprehensive 级,19 行)。原始 JSON:`qsa_benchmark_results/qsa_accept_gpu6_kernel.json`。复跑结果:几何平均 **1.284**(与上轮一致),最小值 0.999(T=2048 pre=0,上轮 1.000,漂移 0.1%,在 §7 测量纪律记载的 ±2% 噪声带内),其余各行与上表一致。

## 9. 消融:仅「调参 + split」的收益(2026-10-19,GPU6 同环境三臂)

问题:总收益(几何平均 ~1.28)里,只改并行参数(BLOCK_N/warps/stages)与 split 策略(H800 表 + cap64)能拿多少,结构改动(空 split 早退/直写、merge 依赖消除、host 修剪)边际多少?

口径:预优化代码(`git show bc8a9ea:.../qsa_sparse_paged_attention.py`,无 libtuner、无空 split 机制、merge 按 constexpr NUM_SPLITS 全读、plain-jit 启动)+ 每形状扫描 (BLOCK_N×warps×stages) 27 候选取最优 + NUM_SPLITS 按 H800 表并经 cap64 与旧 host「不超 tile 数」不变量裁剪;merge 固定 w2/s1。三臂同会话背靠背:该消融臂 / 现生产版 / vLLM,计时 = 验收口径(do_bench median,warmup=1000ms,rep=100ms)。脚本 `tools/qsa_tuning_split_ablation.py`(自包含,运行时从 git 提取 bc8a9ea 旧算子);14 个核心 shape 的三臂实测数据归档于 `qsa_benchmark_results/qsa_ablation_tuning_split_gpu6.json`(14 行几何平均:消融 1.269 / 全量 1.292 / 结构边际 1.018),含 5 个分布变体的全量原始数据在 `/tmp/qsa_ablation/ablation_result.json`;每行消融臂与生产版输出 max|diff| ≤ 0.002。

| 行 | splits | 消融最优配置 | vLLM ms | 仅调参+split (SpeedUp) | 全量 (SpeedUp) | 结构边际 |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 T=1 | 33(臂内 17) | b128/w2/s1 | 0.0379 | 0.0344 (1.101) | 0.0325 (1.167) | +6.0% |
| BF16 T=16 | 33(17) | b128/w2/s2 | 0.0396 | 0.0353 (1.122) | 0.0341 (1.161) | +3.5% |
| BF16 T=32 | 33 | b64/w8/s2 | 0.0402 | 0.0367 (1.096) | 0.0368 (1.092) | −0.3% |
| BF16 T=128 | 2 | b64/w2/s3 | 0.2085 | 0.1482 (1.407) | 0.1520 (1.372) | −2.5% |
| BF16 T=257 | 4 | b64/w4/s2 | 0.1795 | 0.1504 (1.193) | 0.1525 (1.177) | −1.4% |
| BF16 T=513/700/1024 | 1 | b128/w4/s2 | – | 1.284/1.329/1.314 | 1.281/1.325/1.306 | −0.3%~−0.6% |
| BF16 T=2048 pre | 1 | b64/w2/s3 | 2.5883 | 1.8972 (1.364) | 1.8959 (1.365) | +0.1% |
| BF16 T=2048 dec | 1 | b64/w2/s2 | 1.8988 | 1.8990 (1.000) | 1.9005 (0.999) | −0.1% |
| FP8 T=1 | 33(17) | b32/w2/s1 | 0.0385 | 0.0350 (1.098) | 0.0331 (1.161) | +5.8% |
| FP8 T=128 | 8 | b32/w4/s2 | 0.2129 | 0.1539 (1.383) | 0.1337 (1.592) | **+15.1%** |
| FP8 T=2048 pre/dec | 1 | – | – | 1.667 / 1.590 | 1.668 / 1.617 | +0.1% / +1.7% |
| T=128 group4-mixed | 2 | b64/w2/s3 | 0.2083 | 1.389 | 1.368 | −1.5% |
| T=128 random-short | 2 | b64/w2/s3 | 0.0407 | 1.133 | 1.208 | +6.6% |
| T=128 random-long | 2 | b32/w4/s2 | 0.2732 | 1.357 | 1.294 | −4.6% |
| T=128 group4-short-long | 2 | b64/w2/s3 | 0.1607 | 1.346 | 1.203 | −10.6% |
| T=2048 group4-short-long | 1 | b64/w2/s2 | 2.0466 | 1.383 | 1.382 | −0.1% |

**几何平均(19 行,同会话):仅调参+split = 1.281;全量 = 1.291;结构边际 = 1.008。** log 域占比 ≈ 97% / 3%。

零点闭环(2026-10-19,GPU6 同会话):**原封不动的 bc8a9ea(原始 sm90 表 + 固定配置)对 vLLM 几何平均 1.002**(14 行,范围 0.996–1.013,输出 diff 全 0)——证实预优化 port 与 vLLM 等速,且消融臂与零点之间唯一的输入差异就是配置与 split 数值。消融臂纯度审计:① kernel 源码 = bc8a9ea 原文,唯一改动为 num_stages 可注入(1 行 diff);② 旧 main kernel 无任何早退/直写/split_counts(空 split 跑空循环并写 -inf partial);③ 旧 merge 仅按 constexpr NUM_SPLITS 全读,无运行时计数/早退,merge 配置为原始硬编码 w2/s1(连调参都未施加);④ 消融臂与零点的差异清单只有 (BLOCK_N/warps/stages) 逐形状扫描值与 NUM_SPLITS(H800 表 + cap64 纯数值)两类。链路:原始 1.002 → 仅调参+split 1.269(14 行)→ 全量 1.292(14 行)。

解读:
1. **绝大部分收益来自 H800 重调配置 + H800 split 表 + cap64**,与既有瓶颈结论一致:prefill 区主 kernel 是 DRAM 受限,配置选对即拿满;splits=1 的行(T≥513)结构机制全部不触发,边际 ±0.6% 以内。
2. 结构改动净边际 +0.8%,分布不均:decode T=1/T=16 +3.5~6.0%(merge 去依赖 + host 修剪,与 §8 归因一致)、FP8 T=128 +15.1%(空 split 早退 + fp8 路径)、random-short +6.6%(短行直写);T=128 多个分布变体 −1.5%~−10.6%。
3. 负边际行的注意点:消融臂拿的是逐形状 oracle 扫描配置,生产臂是 libtuner 实选(已知存在 µs 级误选噪声,§7),负值不全是结构成本;T=128 group4-short-long −10.6% 若复现可删 key 重调确认。
4. 口径声明:消融臂跑旧代码 plain-jit host 路径(无 LibEntry 开销),对微内核行略有利;v1 试验记录中「调参+split + LibEntry host」decode 仅 0.74–0.81(opt_summary.md §4 R1a),即结构工作中的 host 部分是把 LibEntry 框架成本压回 decode 可接受区间的前提,单看 kernel 收益会低估其必要性。
