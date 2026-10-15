# QSA Sparse Paged Attention 整改结果

日期:2026-10-15。整改起点:`5f61960`;优化前参考:`bc8a9ea`。需求文档:`review.md`。

## 1. 环境与基线溯源

| 项 | 值 |
| --- | --- |
| 验收设备 | NVIDIA H800(sm_90),CUDA 13.1 driver 570.124.06;同机 8 卡中其余卡在整改期间有其它租户负载 |
| torch / triton | 2.11.0+cu130 / 3.6.0(FlagTree) |
| vLLM baseline | **0.31.0 wheel**;导入路径 `/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa.py`(md5 前缀 `0997e4a4e30a`);上游参考 f47792af7f。wheel 无 commit 元数据,以路径+版本+哈希为准 |
| Python / pytest | 3.12.3 / 8.1.1 |
| 计时模式 | kernel(do_bench median,含 L2 flush,受启动间隙影响)/ operator(同步边界端到端 wall)/ cudagraph(预热捕获后重放整条 GPU 链路) |
| `--iter`/`--warmup` | 时间预算(ms)非次数;kernel/cudagraph 用 20/5,operator 用 **200/50**(20 ms 预算下端到端方差过大,统一加大后重测) |

原始数据:`qsa_benchmark_results/qsa_{kernel,operator,cudagraph}_run{1,2}.{txt,json}`(txt 为逐 shape 摘要,json 为框架原始记录,已随仓库归档)。

## 2. R1–R7 整改对照

| 编号 | 问题 | 处理 | 证据 |
| --- | --- | --- | --- |
| R1 | gate 归一化被移除,二维 gate head stride 错误 | 恢复 `output_gate.view_as(q)` 无拷贝归一化;新增 gate 校验(元素数、dtype∈{bf16,fp32}、device、contiguous);gate stride 改为 kernel 内由形状推导(归一化 view 必连续) | `test_gate_shapes`(2D/3D × bf16/fp32 × direct/merge 8 组对拍 vLLM)、`test_gate_invalid`(元素数/dtype/连续性报错) |
| R2 | 私有 plan 缓存持有 Tensor 强引用 | 删除 `_QsaPlan`/`_PLANS`/`_plan_key` 及模板回放;每次调用走 `_QSA_MAIN_TUNED`/`_QSA_MERGE_TUNED`(libentry/libtuner);workspace 每次调用新分配 | `test_lifecycle`(weakref 全释放 + 多签名 live allocation 不累积);删除后量化启动开销见 §4 |
| R3 | 缓存键漏字段、命中后跳过校验 | 校验每次调用执行(键已不存在);合法 dtype/layout 变化经 LibEntry 分派(dtype/对齐在 key 内);`out=` 契约:形状/dtype/device/stride 校验 + **与输入共享 storage 显式拒绝**(autotune 重复执行会污染别名输入);FP8 scale 每次调用现算 | `test_hot_signature_changes`(bf16/fp32 gate、连续/拆分 KV、prefill 交替)、`test_out_contract`、`test_gate_invalid`、`test_empty_rows` |
| R4 | 外层 LibEntry 提前命中,T 分桶不生效 | 新增 `T_BUCKET: tl.constexpr`(host 算 next_pow2),同时进入 LibEntry 外层缓存键与 libtuner key;`num_rows` 保持 dns(按值分桶会爆炸编译数,显式桶只有 log 数量级) | `test_t_bucket_order`(T=33/128 交换首调顺序)、`tests/test_libentry.py::test_libentry_dispatch_by_dtype_and_bucket` |
| R5 | 算子内堆叠逐卡配置表且漂移非目标策略 | 移除 `_select_h800/sm90/sm120_splits` 与设备名分支;split 策略迁入 `_nvidia/heuristics_config_utils.py` 的 `HEURISTICS_CONFIGS["qsa_sparse_paged_attention"]`(经 `runtime.get_heuristic_config` 统一入口);按 capability+设备名区分 h800/sm90/sm120/generic;**恢复 bc8a9ea 漂移**(SM90 FP8 bp64: 16→8、bp128: 8→4、>2048→1;SM120 prefill bp>2048: 1→2 等);缺失配置显式回退 splits=1 | `test_split_policy_static`(monkeypatch 设备类别,对硬编码参考值逐边界核对;策略检查≠性能实测) |
| R6 | NUM_SPLITS/BLOCK_N 与 workspace/merge 耦合 | NUM_SPLITS 在分配与启动前由 host 一次确定;主 kernel 每行写 `split_counts[row]`(0=直写行,否则实际写 partial 的 split 数),merge 只按该值屏蔽读取——不再依赖 BLOCK_N,不存在跨 kernel 猜测;splits 上限 `cdiv(width, _MIN_CANDIDATE_BLOCK_N)` 从 tuner 候选推导,无硬编码常量 | `test_graph_bidirectional`(长↔短/空、tile 边界双向换内容重放)、`test_small_selection_width`(17/63/65 宽度)、`test_t_bucket_order`、原 97 项边界用例 |
| R7 | 性能报告口径混淆、证据不全 | 三模式分开测量、各自 SpeedUp 不跨模式混合;原 14 组保留并补 5 组分布用例(group4/全短/全长/混合,见 `core_shapes.yaml`);每行报告 T/请求数/heads/L 分布/标志;两次独立复测 + 原始 JSON 归档;首次编译/调优与稳定耗时分离记录;`opt_summary.md` 已加历史声明并修正不实表述;5f61960 数字仅作历史对照 | 见 §3、§4 |

## 3. 最终配置与 key

- 主 kernel libtuner key:`T_BUCKET, NUM_QUERY_HEADS, GROUP_SIZE, HEAD_DIM, TOPK, PAGE_SIZE, NUM_SPLITS, stride_k_head, stride_k_token, stride_k_block, stride_indices_row`(KV dtype 随张量 dtype 自动入键);候选 25 组(`tune_configs.yaml: qsa_sparse_paged_attention`)。
- merge kernel key:`NUM_SPLITS(log), HEAD_DIM`;候选 3 组 warp 配置。
- split 策略:`runtime.get_heuristic_config("qsa_sparse_paged_attention")["NUM_SPLITS"]`,H800 BF16 为 H800 实测表(bp≤32→64、≤256→2、≤512→4、>512→1),H800 FP8 与其余设备恢复 bc8a9ea 值。
- NUM_SPLITS 与 workspace/merge 一致性:主 kernel 写 `split_counts`,merge 按值屏蔽(见 R6)。
- 冷启动:tuning 预算 25 ms/100 ms(libtuner 默认);19 个桶全部首轮 tuning 约 5–8 分钟一次性,固化于 LibCache sqlite;**tuning 对机器干扰敏感**(本机整改期间其它 7 卡有租户,pytest 会话内联调曾两次错选 T=700 桶;安静重调后复测恢复,见 §5 风险)**。

## 4. 性能复测(两次独立运行平均;SpeedUp = vLLM / FlagGems;L 分布 mixed = 1/3 行 L=17 其余 ~2048–2051;BF16 K/V 均为非连续 split view)

### kernel 模式(do_bench 口径,含启动间隙影响,≠完整 host wall time)

| Shape | vLLM (ms) | Gems (ms) | SpeedUp | 判定 |
| --- | --- | --- | --- | --- |
| T=1, req=2, Hq=24/Hkv=12/12, prefill=T | 0.0346 | 0.0362 | 0.957 | 减速(如实报告) |
| T=16, req=3 | 0.0363 | 0.0366 | 0.992 | 基本平 |
| T=32, req=5 | 0.0374 | 0.0380 | 0.982 | 基本平 |
| T=128, req=7 | 0.2091 | 0.1532 | 1.365 | 收益 |
| T=257, req=13 | 0.1810 | 0.1542 | 1.175 | 收益 |
| T=513, req=17 | 0.3300 | 0.2589 | 1.275 | 收益 |
| T=700, req=23 | 0.4515 | 0.3406 | 1.325 | 收益 |
| T=1024, req=33 | 1.3010 | 0.9930 | 1.310 | 收益 |
| T=2048, req=63, prefill=T | 2.6052 | 1.9069 | 1.367 | 收益 |
| T=2048, req=63, prefill=F | 1.9137 | 1.9097 | 1.002 | 平 |
| FP8 T=1 | 0.0343 | 0.0343 | 0.999 | 平 |
| FP8 T=128 | 0.2179 | 0.1347 | 1.618 | 收益 |
| FP8 T=2048 prefill | 2.9260 | 1.7410 | 1.680 | 收益 |
| FP8 T=2048 decode | 2.7923 | 1.7479 | 1.598 | 收益 |
| group4 mixed T=128 | 0.2088 | 0.1531 | 1.364 | 收益 |
| 全短 L=17 T=128 | 0.0348 | 0.0342 | 1.015 | 平 |
| 全长 L=2051 T=128 | 0.2767 | 0.2135 | 1.296 | 收益 |
| group4 长短混合 T=128 | 0.1619 | 0.1345 | 1.204 | 收益 |
| group4 长短混合 T=2048 | 2.0432 | 1.4787 | 1.382 | 收益 |

legacy14 几何平均 **1.237**,最低 0.957;BF16 decode 组(T=1/16/32)几何平均 0.977,BF16 大 T 组(128–2048)1.301。

### operator 模式(同步端到端,预算 200 ms/50 ms;run1|run2 差异 ≤2.5%)

| Shape | vLLM (ms) | Gems (ms) | SpeedUp |
| --- | --- | --- | --- |
| T=1 | 0.1307 | 0.1170 | 1.114 |
| T=16 | 0.1022 | 0.1050 | 0.974(减速) |
| T=32 | 0.1014 | 0.1040 | 0.976(减速) |
| T=128 | 0.2028 | 0.1452 | 1.397 |
| T=257 | 0.1737 | 0.1473 | 1.179 |
| T=513 | 0.3232 | 0.2483 | 1.302 |
| T=700 | 0.4429 | 0.3288 | 1.347 |
| T=1024 | 1.2923 | 0.9771 | 1.323 |
| T=2048 prefill | 2.5930 | 1.8968 | 1.367 |
| T=2048 decode | 1.9022 | 1.9004 | 1.001 |
| FP8 T=1 | 0.0915 | 0.0940 | 0.974(减速) |
| FP8 T=128 | 0.2124 | 0.1300 | 1.633 |
| FP8 T=2048 P | 2.8919 | 1.7112 | 1.690 |
| FP8 T=2048 D | 2.7573 | 1.7427 | 1.583 |

legacy14 几何平均 **1.253**,最低 0.974;BF16 decode 组 1.019,大 T 组 1.317。端到端 wall 与 vLLM 同量级(~93–141 µs/次调用,T=1 因 workspace 分配与双 launch 略高)。

### cudagraph 模式(预热捕获后重放,纯 GPU 链路)

| Shape | vLLM (ms) | Gems (ms) | SpeedUp |
| --- | --- | --- | --- |
| T=1 | 0.0085 | 0.0088 | 0.966(减速) |
| T=16 | 0.0162 | 0.0145 | 1.117 |
| T=32 | 0.0268 | 0.0285 | 0.940(减速) |
| T=128 | 0.1990 | 0.1409 | 1.412 |
| T=257 | 0.1703 | 0.1424 | 1.196 |
| T=513 | 0.3201 | 0.2463 | 1.300 |
| T=700 | 0.4404 | 0.3278 | 1.344 |
| T=1024 | 1.2825 | 0.9722 | 1.319 |
| T=2048 P | 2.5860 | 1.8875 | 1.370 |
| T=2048 D | 1.8915 | 1.8894 | 1.002 |
| FP8 T=1 | 0.0115 | 0.0096 | 1.204 |
| FP8 T=128 | 0.2090 | 0.1250 | 1.671 |
| FP8 T=2048 P | 2.9040 | 1.6982 | 1.710 |
| FP8 T=2048 D | 2.7673 | 1.7416 | 1.589 |

legacy14 几何平均 **1.274**,最低 0.940;BF16 decode 组 1.004,大 T 组 1.322。

### 阶段分组与结论

- 分组依据:T(1/16/32 为小批量 decode 形态;128–2048 为 prefill 形态)、请求数、heads、L 分布、布局见上表;`use_prefill_config` 仅作为表中标志列出,不单独作为 prefill/decode 证明。
- **三个模式 legacy14 每组 shape 均 ≥ 0.9**(最低 0.940)。低于 1.0 的项已如实标注为减速/持平:kernel T=1 0.957、T=32 0.982;operator T=16 0.974、T=32 0.976、FP8 T=1 0.974;cudagraph T=1 0.966、T=32 0.940。
- 收益保留:大 T BF16 三模式几何平均 1.30–1.32;FP8 1.58–1.71;新分布用例无未解释回归。
- 对照 5f61960(历史,含不安全私有缓存):其 decode kernel 模式 2.1× 的收益**不可复现也不应复现**;整改后 decode 在 operator 模式 1.114×(T=1)、cudagraph 0.94–1.12×,kernel 模式 0.94–0.99×——小 shape 差异主要来自逐次 Python 启动开销(双 launch 经公共入口)与 vLLM 持平量级。
- 低于 1.1 目标:BF16 decode 组在 kernel 模式几何平均 0.977,未达 1.1 保留收益目标;按 review 完成条件如实记录为**未达标项**(正确性整改优先;公共入口启动开销已量化并做了机制内最小优化,见 `libentry.py` 变更与 `tests/test_libentry.py`,继续压缩需另行评估)。BF16 大 T 组 1.30–1.32 达到 ≥1.1。

## 5. 显存生命周期证据

- `test_lifecycle`:6 组不同签名调用后,`weakref` 确认 op 未保留任何输入/输出;再经 6 轮新签名调用,`active_bytes.all.current` 增长不超过单个 workspace 大小的余量(对照:整改前每个签名保留整套输入+workspace,大 shape KV 约 12.5 GiB)。
- 非 Graph 场景每次调用新分配 workspace(主 partial、LSE、split_counts),无跨调用复用。
- 生产路径审计:op 文件内 `torch.*` 仅为类型注解、dtype 常量与 `torch.empty*` 分配;无 `.contiguous()/.to()/copy_` 等拷贝修补(`.to(` 均位于 Triton kernel 内,为 tl 转换)。

## 6. 测试汇总

- `tests/test_qsa_sparse_paged_attention.py`:155 项(14 回归 + 15 计数模式 × 索引分布 × KV × FP8 边界矩阵 + graph 双向 3 转换 + gate 形状 8 组 + 热签名/空输入/小宽度/T 桶顺序/生命周期/策略静态核对等),整改后全过。
- `tests/test_libentry.py`:4 项公共机制回归(缓存命中换张量、dtype/桶分派、位置参数+默认值、tuple/lambda grid)。
- 命令:`pytest -q tests/test_qsa_sparse_paged_attention.py`(全量)/ `--quick`(快速);benchmark 三模式命令见 `review.md` §6(与本文件 §1 一致)。

## 7. 剩余风险与未测范围

- **tuning 环境敏感性**:共享机器上其它租户负载曾导致两个桶错选(已安静重调恢复);CI/新机器首跑建议在安静环境执行一次全量 benchmark 以固化 LibCache;错选影响限于性能不破坏正确性(mask 逻辑与配置无关)。
- **decode 小 shape**(T=1/32)三模式为 0.94–0.99 的减速/持平,根因是逐次 Python 启动开销(vLLM 同量级);进一步压缩需在公共机制内继续优化(已做:参数元数据缓存、frozenset 成员测试、all_args 位置参数快路径、key 内联,合计 wall 124.6→93 µs)。
- H20/SM120/其它 vendor:仅做了**策略恢复与静态核对**(R5),无这些设备的性能实测,不作性能结论。
- operator 模式 20 ms 预算下端到端方差过大(共享主机),已改用 200/50 预算;跨机器复测应沿用大预算。
- 未扩展项(按 review 范围冻结):indexer/top-k、索引排序、主循环预取、grouped-head merge。
