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

## 3. 组件与敏感度假设

全 FP4 的路径（`engine/model.py`）：

- `_add_full_fp4`：qkv / o（attention）、inp / out（GDN）、gu / down（MLP）都有 MXFP4 副本；
- `_body_full_mxfp4` / `_head_full_mxfp4`：全部投影走 `KQ4.gemm`；
- int8 路径：`_int8_weights` 的 qkv/o/inp/out + `_mxfp4_mlp_weights` 的 MLP FP4；
- 读出头：`_head_full_mxfp4` vs `_head_mxfp4_i8`；
- embedding 在两个路径里都是 int8（Q8）。

假设（待测）：

1. **读出头 / embedding** 对决策最敏感，但 FLOPs 很小；保留 int8 的代价主要是 1GB 显存。
2. **attention qkv/o** 影响证据选择，FP4 误差会直接改路由。
3. **GDN inp/out** 的误差会随状态累积。
4. MLP 已经验证 FP4 无损。

## 4. 配置矩阵

`STARTLUX_PRISM_FP4_KEEP` 是“**保留 int8 的组件**”列表（空 = 全 FP4）：

| 配置 | KEEP | 预计显存 | 相对全 FP4 的速度 | 目的 |
|---|---|---|---|---|
| A | `embed,head` | ~7.0 GB | 几乎不变 | 验证“只保读出头/嵌入”能否恢复质量 |
| B | `attn` | ~7.0 GB | -3~5% | 验证 attention 是否主因 |
| C | `gdn` | ~7.0 GB | -3~5% | 验证 GDN 是否主因 |
| D | `attn,gdn` | ~8.0 GB | 接近 MLP FP4 | 等价于现有 MLP FP4（对照） |
| E | `head` | ~6.5 GB | 几乎不变 | 单独验证读出头 |
| F | `embed` | ~6.5 GB | 几乎不变 | 单独验证 embedding |

可选（时间够再做）：`attn,head`、`gdn,head`、`attn,gdn,head`。

## 5. 引擎改动

1. 在 `Options` / server 增加 `--fp4-keep`（或只读 env `STARTLUX_PRISM_FP4_KEEP`），默认空
   （保持现有全 FP4 行为）。
2. 加载阶段：
   - 先建 int8 骨架 + `_int8_weights(mlp=False)`；
   - 按 KEEP 决定哪些组件保留 int8、哪些用 `_add_full_fp4` 的 FP4 副本；
   - 不保留 int8 的组件在 FP4 副本建好后释放，避免 int8+FP4 双份常驻（16GB 关键）。
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

- 服务器：`--precision mxfp4 --max-tokens 4096 --kv-fp8 --think-seqs 1`，
  `STARTLUX_PRISM_MXFP4_MLP=.../mxfp4/mlp_smooth.pt`，`STARTLUX_PRISM_FP4_KEEP=<配置>`。
- 217 条，`think_mode=auto`，`gate=0.6`，`think_budget=2048`（统一 cap），`return_details`。
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

- `--fp4-keep` / `STARTLUX_PRISM_FP4_KEEP` 实现；
- 一个批量脚本：依次启动各配置、跑 217 条、汇总到一张表；
- 结果写回本文档（第 10 节）和 `docs/fp4.md`；
- 最终建议：是否用混合 FP4 替换现有 MLP FP4；是否值得进入 QAD。
