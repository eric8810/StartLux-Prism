# 混合 FP4 灵敏度实验设计（2026-10-25）

## 1. 要回答的问题

全 FP4 的 217 条只有 **56.2%**（int8 / MLP FP4 都是 **66.8%**），差 **10.6 分**。在不做 QAD
的前提下，能否通过“只把敏感组件保留 int8/FP8”把质量拉回 **≥65.5%**，同时保留全 FP4 的
大部分速度和显存收益？

同时要区分两个数字：

- **速度收益**：全 FP4 + CUTLASS 相对 int8 的 1.6–1.7x（1k 纯读入的估算，非已测）。
- **质量恢复**：把 10.6 分差距补回多少；补回 60–70% 只有 62.6–63.6%，仍不可用，需要补回 90%+。

## 2. 已知基线（217 条，auto，gate 0.6）

| 配置 | 显存（1k） | gold | 1k 延迟 | 说明 |
|---|---|---|---|---|
| int8 | 11.3 GB | 66.8% | 183.4 ms | 参考 |
| MLP FP4（平滑，单份） | 8.0 GB | 66.8% | 146.5 ms | 当前推荐 |
| 全 FP4（PTQ） | ~6.0 GB | 56.2% | 126.2 ms | 质量不可用 |
| 全 FP4 + CUTLASS | ~6.0 GB | 待测 | 估计 90–110 ms | 本实验的目标形态 |

最新（32k、FP8 KV、CUTLASS）的同条件数字：1.8k 读入 192.6 ms（int8 272.6），32 条短读入
平均 89.9 ms（Triton FP4 104.8）；这些是 MLP FP4，不是全 FP4。

## 3. 组件与敏感度假设（按代码实际路径修正）

`_add_full_fp4` 只把 **attention 的 qkv/o** 和 **GDN 的 inp/out** 换成 MXFP4；`gu/down` 在
MLP FP4 里已经是 FP4。因此真正的变量只有两个：

- **attention（qkv/o，8 层）**
- **GDN（inp/out，24 层）**

不参与量化的部分：

- embedding：所有路径都是 int8（Q8）；
- 读出头（`out_norm` + `letters`）：bf16/fp32；
- MLP：所有 mxfp4 配置都是 FP4；
- KV：单独的 FP8 缓存。

所以“保留 embed/head”不是独立开关；`_head_*` 里的 qkv/o 属于 attention，最后两层 MLP 属于
MLP。全 FP4 的 56.2% 损失只可能来自 attention 和/或 GDN。

假设：

1. attention 的 FP4 误差直接影响证据选择/路由，可能更敏感；
2. GDN 的 FP4 误差会随 24 层状态累积，可能更敏感；
3. 二者按层分布可能不均匀，必要时做“部分层 FP4”。

### 3.1 思考解码的约束

`engine/decode.py` 读的是 int8 的 `qkv_q/o_q/inp_q/out_q`（思考解码仍走 W8A8 attention/GDN +
FP4 MLP）。因此：

- prefill 用 FP4 的组件，若还要支持思考，**必须保留它的 int8 副本**；
- 带思考时，FP4 组件的显存收益会被 int8 副本吃掉（每个组件约 0.5GB）；
- 第一次扫描用 **`think_mode=never` + `--think-seqs 0`**，只量 prefill 的质量和显存；
  选定配置后再用 `auto` 测思考行为，并单独记录带 int8 副本的显存。

全 FP4 的 56.2% 是 auto 下测的（解码器是 int8）；扫描时要在 never 下重测所有配置，避免把
“prefill 的 FP4 误差”和“int8 解码器的补偿”混在一起。

## 4. 配置矩阵

环境变量：

- `STARTLUX_PRISM_FP4_ATTN=1`：attention qkv/o 用 FP4；
- `STARTLUX_PRISM_FP4_GDN=1`：GDN inp/out 用 FP4；
- 两者都不设 = 现有 MLP FP4（int8 attention/GDN）；
- 两者都设 = 全 FP4；
- `STARTLUX_PRISM_FULL_FP4=1` 保留为“两者都设”的快捷方式。

| 配置 | attention | GDN | 预计显存（1k） | 质量 | 目的 |
|---|---|---|---|---|---|
| A | int8 | **FP4** | ~7.5 GB（never） | 待测 | GDN 是否主因 |
| B | **FP4** | int8 | ~7.5 GB（never） | 待测 | attention 是否主因 |
| C | int8 | int8 | 8.0 GB（never） | 66.8% | 现有 MLP FP4（对照） |
| D | **FP4** | **FP4** | ~6.5–7.0 GB（never） | 56.2%（auto，需在 never 复测） | 全 FP4（对照） |

第一轮 A/B 就能定位主因。若 A 或 B 单独接近 66%，再测：

| 配置 | 说明 |
|---|---|
| E | attention 只有前 N 层 FP4（其余 int8） |
| F | GDN 只有前 N 层 FP4（其余 int8） |
| G | 主因组件在所有层 FP4，另一组件只在部分层 FP4 |

显存估算的基线：int8 attention/GDN 各约 0.5GB，FP4 版各约 0.25GB；全 FP4 比 MLP FP4 少约
1GB，所以是 7.0–8.0GB 区间，不是 6GB（6GB 是另一条加载路径）。

## 5. 引擎改动

1. 增加 `STARTLUX_PRISM_FP4_ATTN` / `STARTLUX_PRISM_FP4_GDN`（只读 env 即可，实验阶段不加
   server flag）；默认都不设，保持现有 MLP FP4 行为。
2. 加载阶段：
   - 先建 int8 骨架 + `_int8_weights(mlp=False)`；
   - 按两个开关决定哪些组件用 `_add_full_fp4` 的 FP4 副本；
   - 用 FP4 的组件不再构建 int8 副本（或建好后立即释放），避免双份常驻（16GB 关键）。
   - MLP 始终 FP4（`_mxfp4_mlp_weights`）。
3. 前向：新增混合 body/head（或在现有两个 body 里按组件选 kernel）：
   - attention：int8 `KQ8.gemm` 或 FP4 `KQ4.gemm`；
   - GDN：int8 或 FP4；
   - MLP：FP4；
   - head：int8 或 FP4；
   - embedding：int8（Q8）。
   - 每个投影的输出 stride、epilogue（epi 1 的 fp32 残差）与现有路径一致。
4. CUTLASS：质量阶段可以先关（`STARTLUX_PRISM_CUTLASS` 不设），只用 Triton FP4，避免把
   kernel 差异混进质量对比；选定配置后再开 CUTLASS 测速度。CUTLASS 与 Triton 已证明逐位一致，
   所以质量结论可以沿用。
5. 显存：加载时按组件顺序构建/释放，并在启动日志打印最终各组件精度和峰值。

## 6. 评估协议

- 第一阶段（质量/显存）：`--precision mxfp4 --max-tokens 4096 --kv-fp8 --think-seqs 0`，请求 `think_mode=never`；
  第二阶段（思考行为）：选定配置后用 `--think-seqs 1` 和 `think_mode=auto`。
  `STARTLUX_PRISM_MXFP4_MLP=.../mxfp4/mlp_smooth.pt`，`STARTLUX_PRISM_FP4_ATTN=<0/1>`，
  `STARTLUX_PRISM_FP4_GDN=<0/1>`。
- 217 条；第一阶段 `think_mode=never`，第二阶段 `think_mode=auto`，`gate=0.6`，`think_budget=2048`，`return_details`。
- 记录：gold、thought 条数/token、P50/P95 延迟、启动显存、峰值显存。
- 参考：int8（66.8）、MLP FP4（66.8）、全 FP4（56.2）。
- 指标：gold/217；相对 MLP FP4 的 mean |Δp|；相对 int8 的 mean |Δp|；思考行为（条数/token）。
- 噪声：217 条约 ±3 分。**65.5–66.8 之间的差异需要 500–600 条复测**才能定论；第一阶段先看
  是否 ≥65.5 和思考行为是否接近。

## 7. 判定标准

一个配置算“可用”，需要同时满足：

1. gold **≥65.5%**（217 条初筛；≥65.5 后扩到 500+ 条确认）；
2. 思考条数/token 与 MLP FP4 相差 **≤10%**（AutoThink 行为不能明显改变）；
3. 显存 **≤8 GB**（1k）；
4. 速度 **≥1.2x MLP FP4**（CUTLASS 开）。
5. 217 条相对 MLP FP4 的 mean |Δp| **≤2%**（概率层面不出现大漂移）。

若 A 达标，它就是“接近全 FP4 速度 + 质量可用”的答案，不需要 QAD；若 A/B/C/E/F 都不达标，
说明损失分散，QAD 才值得考虑。

## 8. 成本与风险

- 代码：4–8 小时（混合加载 + 前向选择 + 日志）。
- 评估：每个配置 10–20 分钟；6 个配置约 **1.5–2 小时**；扩到 500 条再约 1 小时/配置。
- 总时间：**1–2 天**，不需要训练。
- 风险：
  - 加载时 int8+FP4 双份可能 OOM → 按组件顺序构建并立即释放；
  - 全 FP4 的损失可能分散，所有配置都到不了 65.5；
  - 思考行为改变（参考 MXFP4 KV 的教训）；
  - CUTLASS 在全 FP4 的更多形状上未验证（质量阶段先不用它）。
- 不做：QAD（需要训练数据/teacher/云端 GPU，数天到数周）。

## 9. 交付物

- `STARTLUX_PRISM_FP4_ATTN` / `STARTLUX_PRISM_FP4_GDN` 实现；
- 一个批量脚本：依次启动各配置、跑 217 条、汇总到一张表；
- 结果写回本文档（第 10 节）和 `docs/fp4.md`；
- 最终建议：是否用混合 FP4 替换现有 MLP FP4；是否值得进入 QAD。

## 10. 第一轮结果（2026-10-25）

### never（234 条，同条件）

| 配置 | gold | 同选择 | mean \|Δp\| |
|---|---|---|---|
| C：MLP FP4 | 133/234 = 56.8% | — | — |
| A：GDN FP4 | 127/234 = 54.3% | 192/234 | 6.27% |

首轮读入只差 **2.5 分**，GDN FP4 不是首轮主因。

### auto（2048 cap，同一组 217 条）

| 配置 | gold | 思考条数 / token | 平均延迟 |
|---|---|---|---|
| C：MLP FP4（旧基线） | **145/217 = 66.8%** | 76 / 53,030 | 5.96 s |
| A：GDN FP4 | **137/217 = 63.1%** | **105 / 107,291** | **11.0 s** |

A 比 C **低 3.7 分**，思考条数 +38%，思考 token **2.0x**，平均延迟 **1.85x**。

结论：

- GDN FP4 在首轮只有 -2.5 分，但在 AutoThink 下放大到 -3.7 分，并把思考量翻倍；
- 这是“行为改变”，不是单纯的分数噪声；即使分数接近，吞吐也明显变差；
- 旧的“全 FP4 56.2%”比 A 还低 6.9 分，剩余差距很可能来自 attention FP4（尚未测）；
- A 未达到 65.5% 的可用线，**GDN FP4 不可用**；
- 下一步：打包 attention qkv/o 的 FP4（`full_smooth.pt` 没有）测 B；如果 B 也明显掉分，全 FP4
  的 PTQ 路线基本关闭，只能走 QAD 或保持 MLP FP4 + CUTLASS。

C 是旧的 `dp_b_smooth8`（auto，2048，217 条）；A 是 4k/FP8 KV。KV 格式的已知差异约 0.4 分，
不影响 3.7 分的结论；必要时再用当前配置重跑 C。

### 当前推荐

- 生产仍是 **MLP FP4 + CUTLASS**（66.8%，1.15–1.17x，bit-identical）；
- 不要开 `STARTLUX_PRISM_FP4_GDN` / `_ATTN`，除非后续 B 能到 65.5%+。

### 第二轮：B（attention FP4，GDN int8）

`full_smooth.pt` 没有 attention 的 qkv/o，加载时用 Q8_0 权重 + int8 的 SmoothQuant scale
（`W * s`，`s = attn_norm / attn_norm_s`；o 用 `1 / o_inv`）现场打包。

| 测试 | B（attention FP4） | C（MLP FP4） |
|---|---|---|
| never，234 条 | 127/234 = **54.3%** | 133/234 = 56.8% |
| auto，217 条 | **138/217 = 63.6%** | 145/217 = 66.8% |
| 思考条数 / token | **99 / 109k** | 76 / 53k |
| 平均延迟 | **11.2 s** | 5.96 s |

B 和 A（GDN FP4）在 never 下同为 54.3%，auto 下 B 63.6% / A 63.1%，思考量都约 2x。
之前的“B vs A max |Δp| = 0”是评分脚本的空字典 bug；逐条直接对比概率是明显不同的。

## 11. 结论（2026-10-25）

- 单独把 **GDN** 或 **attention** 换成 FP4，首轮只掉 2.5 分，但 AutoThink 下都掉 3.2–3.7 分，
  并把思考量翻倍（76/53k → 99–105 / 107–109k），延迟 1.85–1.9x。
- 两个组件都保留 int8 才是 66.8%；任何一个换 FP4 都过不了 65.5% 的可用线。
- 因此 **全 FP4 的 PTQ 路线关闭**（旧的 56.2% 与“两者都掉 + 行为放大”一致），
  除非做 QAD，否则生产继续用 **MLP FP4 + CUTLASS**（66.8%，1.15–1.17x，bit-identical）。
- QAD 仍是唯一可能拿到全 FP4 的路线，但需要训练数据/teacher/云端 GPU；本次实验不再继续。
