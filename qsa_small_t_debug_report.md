# QSA 小 T 性能调试报告(证据轮)

日期:2026-10-16。代码基点:`3e60d06`。依据:`qsa_small_t_debug_guide.md`。
本轮**只采集证据,未修改生产代码**。唯一代码变更是修复采集脚本本身
(`tools/qsa_small_t_debug.py` 交付时缺前半部分,补齐后按指南命令运行)。

## 1. 执行记录

- 环境:单机 8×H800,采集期间**全部空闲**(无其它租户);GPU 0;torch 2.11.0+cu130、triton 3.6.0、vLLM 0.31.0。
- 命令:指南 §2 四条(run1、run2、profile_fp8_t1、profile_bf16_t32)+ §7 两条控制实验(short、group4)。全部一次成功,无精度失败。
- 两轮(run1/run2)launch 配置逐项相同(脚本 results.json 对比一致);延迟波动多数 <5%。
- 原始证据:`qsa_small_t_results/`(summary.csv、results.json、profile 的 trace/cpu/gpu/python/pstats,及整包 tarball)。

## 2. 计时汇总(allocate 模式,两轮中位数平均;SpeedUp=vLLM/Gems)

| 案例 | operator | kernel | cudagraph |
| --- | --- | --- | --- |
| BF16 T=1 (L=2051) | 0.982 | 0.969 | 0.991 |
| BF16 T=16 | 0.982 | 0.982 | 1.134 |
| BF16 T=32 | 0.993 | 0.994 | **0.944** |
| FP8 T=1 | 0.975 | 0.993 | **1.269** |
| FP8 T=16 | 0.993 | 1.006 | 1.204 |
| FP8 T=32 | 0.986 | 1.470 | 1.758 |
| 控制:short(L=17) T=1 / T=32 | 0.980 / 0.978 | – | 1.43 / **2.14**(BF16)、1.66 / 1.81(FP8) |
| 控制:group4 T=1 / T=32 | 0.988 / 0.983 | – | 1.00 / 0.94(BF16)、1.24 / 1.70(FP8) |

- operator 模式所有案例 Gems 慢 1–2.5 µs(固定差距,与 T/dtype 无关)→ host 侧固定开销差。
- preallocated out 对照:vLLM 省 ~3.5–4 µs,Gems 只省 ~0.7 µs → Gems 的 out= 路径新增 alias 检查约 3 µs(7×untyped_storage);未预分配时两者差距约 1.5–2.5 µs。
- 全部案例数值与指南 §1 引用的历史两轮一致(decode operator ~0.97–0.99)。

## 3. 证据分析(按指南 §6 三问)

### A. host(FP8 T=1:Graph 快 1.27×、operator 慢 0.975× → host 吃掉了 GPU 收益)

`.python.txt`(200 次调用,cProfile self time 排序)与 `.cpu.txt`:

| 项 | Gems | vLLM | 差 |
| --- | --- | --- | --- |
| 算子 wrapper | ~20 µs(profiled) | ~25 µs | Gems 更省(heuristics/alias 跳过等) |
| dispatch(LibEntry.run×2 vs jit.run×2) | ~35 µs | ~25 µs | **Gems +~10 µs(profiled)≈ +1–2 µs 真实** |
| 设备查询(_qsa_device_class+get_device_name/capability) | ~5 µs | – | 次要(~1 µs) |
| workspace 分配 | empty ×3(含 split_counts) | empty ×2 | **+~1.5 µs** |
| alias 检查(仅 out= 传入) | 7×untyped_storage | 无 | +~3 µs(preallocated 模式下显形) |

结论:host 差距 ≈ LibEntry 参数整理/缓存查找(公共机制,已做过一轮最小优化)+ 一个 split_counts 分配 + out= 的 alias 检查;合计与 operator 模式实测差(1.5–2.5 µs,preallocated 时 ~4.5 µs)吻合。`_select_splits`/heuristics 调用本身 ~1 µs,次要。

### B. GPU main vs merge(profile,.gpu.csv,5 次调用中位)

| 案例 | main vLLM | main Gems | merge vLLM | merge Gems |
| --- | --- | --- | --- | --- |
| BF16 T=32 | 24.86 µs | 24.86 µs | 3.68 µs | **6.08 µs** |
| FP8 T=1 | 10.98 µs | **6.72 µs** | 2.88 µs | **4.07 µs** |

- **BF16 T=32 的 cudagraph 劣势(0.944)全部来自 merge(+2.4 µs);main 与 vLLM 完全相同**(含 64 vs 33 split 启动、31 个空 split 早退——这些在 main 里免费)。
- FP8 T=1:main 快 4.3 µs(R2 空 split 早退 + 更优配置:vLLM fp8 主 kernel 有 72 寄存器 spill,Gems 无),merge 慢 1.2 µs,净胜 2.4 µs。
- merge 变慢的机制:Gems merge 的读取掩码依赖运行时 `split_counts`(标量加载→两个向量 load 发射等待),且掩码是运行时的,丧失 vLLM 编译期 `NUM_SPLITS` 上界的向量化;vLLM merge 自 launch 起即可全速发射 load。

### C. split 数量与配置

- 实际 launch(run1,两轮一致):BF16 T=1/16/32 Gems `S=64`(grid 第 3 维 64,其中 tiles(BLOCK_N=64)=33 之外的 31 个空 split 早退)、vLLM `S=33`;FP8 T=32 双方 `S=16`(Gems b32w4 无 spill vs vLLM b32w1 spill=128)。
- short 控制(L=17):cudagraph 下 Gems 1.4–2.1× —— R2 的"空 split 早退 + 单 tile 直写"收益确认;operator 仍 ~0.98(host 主导)。
- group4 控制:与 random 几乎相同(BF16 T=32 均 0.94)→ **索引局部性不是本算子的因素**(gather 以 512B token 行为粒度,4-token 分组不改变 DRAM 效率);无需为索引排序投入。

## 4. 结论与下一轮建议(未实施)

按指南 §8 决策表映射:

1. **优先:merge 的 split_counts 依赖(直接解释 BF16 T=32 的 0.94 与全部案例 merge +1.2~2.4 µs)**。候选(下一轮按"main+merge 整体"评估,不只看 main):
   - 空 split 退出前写 `-inf` LSE(仅 GROUP_SIZE 个标量,保留 Q 加载跳过),merge 回到编译期 `NUM_SPLITS` 掩码;partial_output 仍需处理脏读(NaN×0 问题),可能仍需对 partial 读取保留运行时掩码或用"写零 partial"换回(注意 short 场景回退,需实测)。
   - 或维持 split_counts 但消除发射依赖(实验性,收益不确定)。
2. **次优先:host 固定开销(~1.5–2.5 µs)**:LibEntry 属公共机制,继续优化需走公共路径+回归;split_counts 可与 LSE 同 Buffer 分配(省一次 empty);out= 的 alias 检查可用 data_ptr 区间比较替代 untyped_storage(省 ~2 µs,仅 out= 路径)。单项收益均 <1–2 µs,属锦上添花。
3. **无需动作**:FP8 T=32 收益(1.47–1.76)已证明 autotune 配置质量价值;索引局部性已排除;split 早退在 main 侧免费。

## 5. 交付物

- `tools/qsa_small_t_debug.py`(修复后可运行;与生产代码分离)
- `qsa_small_t_results/`(run1/run2/short/group4 的 summary.csv+results.json,两个 profile 目录的全部 trace 与聚合表,整包 tarball)
- 本报告
