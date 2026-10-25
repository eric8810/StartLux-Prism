# 上下文显存：现状、优化路线与场景预估

记录时间：2026-10-25。对应配置：StartLux-Decision-v1.1-9B-AutoThink，RTX 5060 Ti 16GB（15.5 GiB 可用）。

## 1. 当前基线（已落地）

| 项 | 值 |
|---|---|
| 精度 | MLP MXFP4（平滑 alpha 0.5）+ attention/GDN/embed int8 |
| 权重 | 单份，不再加载 int8 MLP |
| 显存 @2k | 8.0 GB |
| 质量（217 条） | 66.8%（与 int8 同分，int4 64.1%） |
| 1k 延迟 | 146.5 ms |
| 启动 | `--precision mxfp4 --calib int8_calib.pt`，`STARTLUX_PRISM_MXFP4_MLP=$M/mxfp4/mlp_smooth.pt` |

实测 `--max-tokens` 与显存：

| max-tokens | 显存 | 结论 |
|---|---|---|
| 2,048 | 8.0 GB | 充足 |
| 4,096 | 8.7 GB | 充足 |
| 8,192 | 10.0 GB | 可用 |
| 16,384 | 12.1 GB | 推荐 |
| 24,576 | 14.1 GB | 实用上限（分块前） |
| 28,672 | 15.1 GB | 极限（分块前） |
| 32,768 | — | 分块前 OOM |

**MLP token 分块已落地**（`STARTLUX_PRISM_MLP_CHUNK`，默认 4096）：

| max-tokens | 分块前 | 分块后 |
|---|---|---|
| 24,576 | 14.1 GB | 约 12.3 GB（估算） |
| 32,768 | OOM | **13.8 GB（实测）** |
| 40,960 | OOM | **15.1 GB（实测，极限）** |
| 49,152 | OOM | 预计 OOM |

正确性：同一 40 条 never 测试 26/40，与分块前一致；17.8k 请求 2.91s（分块前 2.88s），
8.9k 1.31s（前 1.29s），分块开销可忽略。分块只影响 MLP 的中间缓冲，不改每行计算。

长上下文延迟（think_mode=never）：

| 输入 | 延迟 | 吞吐 |
|---|---|---|
| 2.3k | 298 ms | 7,723 tok/s |
| 4.5k | 615 ms | 7,340 tok/s |
| 8.9k | 1,292 ms | 6,922 tok/s |
| 17.8k | 2,878 ms | 6,187 tok/s |

## 2. 每 token 的增长（256 KB）

实测 8k→16k→24k 每 8k 增加约 2.0–2.1GB，即 256KB/token。

| 项 | 每 token | 32k 时 |
|---|---|---|
| KV：主序列（8 层 full attention，4 KV head × 256 × K/V × bf16） | 32 KB | 1.07 GB |
| KV：思考预留（按 max-tokens 留一整份） | 32 KB | 1.12 GB |
| MLP 缓冲：`gu` 48 + `act` 24 + `x` 8 + 量化 9 | 89 KB | 2.98 GB |
| `proj` [T, 12352] bf16 | 25 KB | 0.81 GB |
| `h` + `o`（fp32） | 33 KB | 1.08 GB |
| `aq`（int8） | 12 KB | 0.40 GB |
| `mix` + q/k/v/qbuf | 24 KB | 0.80 GB |
| 其他 | 9 KB | 0.30 GB |
| **合计** | **256 KB** | **约 8.0 GB** |

固定部分约 7.9GB：权重 6.4（FP4 MLP 2.4 + int8 attn/GDN 2.0 + embed 1.0 + 思考 head 1.0）+ CUDA/allocator/图池约 1.5。

**KV 只占增长的 26%**；其余 74% 是按 `max-tokens` 一次性分配的激活缓冲，和请求实际长度无关。

## 3. 四项优化

| 优化 | 省 / token | 说明 |
|---|---|---|
| **MLP token 分块（已落地，CHUNK=4096）** | −89 KB | 逐 token 独立，不改数学；另加 0.36GB 固定缓冲。参考 vLLM chunked prefill、llama.cpp `-ub` |
| 思考 KV 按实际 prompt+思考预算预留 | 0–0.7 GB | 不必等于 max-tokens；取决于实际最大输入 |
| **FP8 KV（`--kv-fp8`，已实现）** | −16 KB/行 | 省 0.6GB@32k，66.4%（bf16 66.8）；推荐 |
| `h`/`o` 改 bf16 | −16 KB | `h` 是残差主干，风险较高；只改 `o` 更稳 |

四项后：固定约 7.9GB（+0.36GB 分块缓冲），增长 **118 KB/token**。

参考实现：vLLM `--enable-chunked-prefill` / `--max-num-batched-tokens`、llama.cpp `-ub`、SGLang `--chunked-prefill-size`、TensorRT-LLM `enable_chunked_context`；融合 SwiGLU 参考 vLLM `SiluAndMul`、TensorRT-LLM gated MLP、Liger/xformers。

### 3.1 KV 格式对比（统一 2048 cap，217 条）

| KV 格式 | 参数 | gold | 思考条数 | 思考 token | 32k 显存 |
|---|---|---|---|---|---|
| bf16 | 默认 | 66.8% | 76 | 52,030 | 13.8 GB |
| **FP8 e4m3** | `--kv-fp8` | **66.4%** | 76 | 53,303 | **13.2 GB** |
| int8 | `--kv8` | 65.0% | 76 | 54,525 | 13.2 GB |
| MXFP4 | `--kv4` | 63.1% | 62 | 42,129 | 12.9 GB |

结论：

- **FP8 e4m3 几乎无损**（-0.4 分，噪声内），思考行为与 bf16 一致，省 0.6GB → 推荐。
- int8 再低 1.4 分，相对 FP8 没有优势。
- MXFP4 低 3.7 分，且改变思考行为（62 条、42k token）→ 不建议用于决策。
- 更正：早先 int8 的“思考 3 倍”是 **4096 cap** 造成的，不是格式；统一 2048 后所有 1 字节格式的思考量与 bf16 持平。

`o`（GDN 输出）改 bf16 省 0.27GB，但组合评估降到 62.7%（单看可能是噪声，但收益太小）；
已回退，保持 fp32。

**最终配置：MLP 分块 + FP8 KV + bf16 `o` 保持 fp32 → 32k 13.2GB，66.4%。**

## 4. 场景预估（四项优化后）

| 显卡 | 显存 | FP4 原生 | 开思考上限 | 关思考上限 |
|---|---|---|---|---|
| 5060 Ti | 16GB | 是（sm_120） | 48–56k | 70–78k |
| 5090 | 32GB | 是（sm_120） | 约 190k | 约 230k |
| RTX 6000 Ada | 48GB | 否（sm_89，模拟） | 约 330k | 约 390k |
| H100 | 80GB | 否（模拟） | 约 600k | 约 700k |
| RTX PRO 6000 Blackwell | 96GB | 是（sm_120） | 约 735k | 约 860k |
| H200 | 141GB | 否（模拟） | 约 1.3M | 约 1.3M |
| B200 | 192GB | 否（模拟） | 约 1.8M | 约 1.8M |

1M 的门槛：1.01M × 118KB ≈ 119GB，加固定约 127GB（开思考）；关思考约 110GB。所以 **96GB 卡跑不了 1M**，需要 H200/B200 级别。

关思考只是容量参考：官方模型卡写明 DecisionPivot 上思考 59.9%、关思考 22.3%，不是可用配置。

## 5. 模型原生上下文与 YaRN

- 配置与 GGUF 都是 `max_position_embeddings = 262144`（256k），原生。
- 官方 Qwen3.5-9B 模型卡：**262,144 原生，可扩展到 1,010,000**，方式是 YaRN：
  `rope_type: "yarn"`、`factor: 4.0`、`original_max_position_embeddings: 262144`、保留 mrope 配置。
- 当前 checkpoint 是 `rope_type: "default"`；Prism 没有 YaRN 实现。
- 256k 是真实需求：96GB 卡上约 38GB，可行；16GB 卡不可行。
- YaRN 是 static scaling，官方提醒会影响短文本；目标 524k 时 factor 用 2.0。

## 6. KV 分层到 RAM/SSD

买的是**并发/缓存容量**，不是单请求长度：

- 单请求 KV 必须在 GPU，或按页流式换入；PCIe 带宽比 HBM 低 50–100 倍，交互式不可行。
- 62GB RAM：256k bf16 KV ≈ 8GB → 约 7 份；FP8 → 约 15 份。
- 1TB SSD：约 125 份 256k bf16；8GB 换入约 1.1 秒（7GB/s）。
- 这个模型 32 层里 24 层是 GDN，recurrent state 固定约 4MB/层（约 100MB/序列），只有 8 层有 per-token KV；换出/换入主要搬 KV。
- 前缀缓存（同一长文档共享）是最值钱的并发手段；当前引擎只有 2 条前缀缓存。

参考：vLLM CPU KV offload、LMCache、Mooncake、InfiniGen。

## 7. 市场参考（OpenRouter，进行中）

2026-10-25 拉取 `https://openrouter.ai/api/v1/models`：467 个模型。

| 上下文 | 模型数 |
|---|---|
| <16k | 10 |
| 16–32k | 3 |
| 32–64k | 17 |
| 64–128k | 10 |
| 128–256k | 122 |
| 256k–1M | 111 |
| ≥1M | 194 |

决策/分类类模型（OpenRouter 描述里明确写 classification / safety / decision / router）：

| 模型 | 上下文 | 输入价 ($/M) |
|---|---|---|
| openai/gpt-6-luna（classification, high-volume） | 1,050,000 | 0.10 |
| openai/gpt-5.6-luna | 1,050,000 | 0.20 |
| openai/gpt-oss-safeguard-20b（safety classification） | 131,072 | 0.075 |
| meta-llama/llama-guard-4-12b（content safety） | 163,840 | 0.18 |
| **typesafe/jev-router**（System One 路由/决策） | **1,000,000** | router 计价 |

结论：**规格层面**，决策/分类模型已经是 128k–1M，不是 10–20k；TypeSafe 自己的 JEV 也是 1M。
“10–20k”是**实际 workload** 的数量级，不是模型上限。

实际请求（OpenRouter 模型页内嵌的 usage 统计：请求数、prompt tokens、cached tokens）：

| 模型 | 类别 | 总请求数 | 平均 prompt | 缓存占比 | **平均新鲜输入** |
|---|---|---|---|---|---|
| inception/mercury-decide:free | 决策 | 26,370,246 | 4,462 | 0% | **4,462** |
| meta-llama/llama-guard-4-12b | 安全分类 | 8,060,558 | 881 | 0% | **881** |
| openai/gpt-oss-safeguard-20b | 安全分类 | 10,052,979 | 1,252 | 64.9% | **440** |
| openai/gpt-5.6-luna | 分类 | 821,071,531 | 16,534 | 77.5% | **3,716** |
| openai/gpt-6-luna | 分类 | 1,587,037,590 | 21,820 | 81.5% | **4,045** |
| **typesafe/jev-router**（System One 路由） | 决策 | 9,440,576 | 11,848 | 74.7% | **3,003** |

结论：

- 决策/分类模型的实际**总 prompt** 约 0.9k–22k；TypeSafe 自己的 JEV 路由是 **11.8k**，正好落在“10–20k”区间。
- 扣掉前缀缓存后，**真正需要计算的新鲜输入只有 0.4k–4.5k**；JEV 是 **3k**。
- 所以“10–20k”是对**总上下文**的直觉，“新鲜输入”更小。256k 是规格/竞争项，不是决策 workload 的刚需。

实际决策输入（DecisionPivot 217 条，`input_tokens`）：

| 统计 | tokens |
|---|---|
| 中位数 | 988 |
| p90 | 1,783 |
| p95 | 1,897 |
| p99 | 2,066 |
| 最大 | 2,077 |

加上思考（README：DecisionPivot 平均约 1,365 thought tokens/题），单题实际消耗约 2.3k tokens。
所以对本模型：**4–8k 上下文覆盖 100% 的基准输入**，16–32k 覆盖更长工单；256k 是规格/竞争项，不是决策 workload 的刚需。

### 7.1 平均输入 vs 最大输入（需求定义）

四家决策模型的实际 usage（OpenRouter 模型页）：

| 模型 | 最大输入 | 平均输入 | 请求数 | 缓存 | 输入价 $/M |
|---|---|---|---|---|---|
| **cloudflare/clef-flash**（9B，Qwen3.5 微调） | **24,576** | **2,075** | 52.5M | 0% | 0.019–0.038 |
| inception/mercury-decide | 65,536 | 4,462 | 26.4M | 0% | 0.02（有免费档） |
| perplexity/pplx-decider-v1.2 | 262,144 | 1,752 | 103.9M | 0% | 0.02 |
| typesafe/jev-1.14 | 262,144 | 3,714 | 500.1M | 0% | 0.042 |
| openai/gpt-6-luna-decisions | 1,050,000 | 2,919 | 85.0M | 0% | 0.10 |
| typesafe/jev-router（路由） | 1,000,000 | 11,848（75% 缓存 → 3,003 新鲜） | 9.4M | 74.7% | 路由计价 |
| openai/gpt-6-luna（分类） | 1,050,000 | 21,820（82% 缓存 → 4,045 新鲜） | 1.59B | 81.5% | 0.10 |

结论：

1. **平均输入非常一致：1.75k–4.5k**，与最大上下文无关。连 1.05M 上下文的 luna-decisions，平均也只有 2.9k。
2. **最大输入分档**：24k（9B 的 clef-flash）→ 65k（mercury-decide）→ 262k（pplx/jev-1.14）→ 1.05M（luna-decisions）。
3. **同类对照**：同样是 Qwen3.5-9B 微调的 clef-flash，最大输入就是 **24k**，平均 2k。这正是 16GB 卡现在的 24k 工作点。
4. 决策专用模型（clef/mercury/pplx/jev-1.14/luna-decisions）缓存都是 0%；只有通用分类流量（luna、jev-router）才有 75–82% 缓存，扣掉缓存后的新鲜输入仍是 3–4k。

需求定义建议：

| 档位 | 最大输入 | 平均输入 | 覆盖场景 | 硬件 |
|---|---|---|---|---|
| 核心决策（9B 同级） | **24–65k** | 2–4k | 工单/状态/多图决策 | 16GB（24k）或 32GB（190k 余量） |
| 长文档决策 | 262k | 2–4k（文档走缓存/截断） | 合同、报告、长记录 | 32GB–96GB |
| 文档/agentic 旗舰 | 1M | 2–4k | 超长文档、跨文档 | 96GB+ / H200 |

对本模型的落地：**16GB 卡按 24k 部署就等于 clef-flash 的规格**；四项优化后 48–56k，覆盖到 mercury-decide 的 65k 档；262k/1M 属于长文档档，不是核心决策需求。

## 8. 待办

1. ~~实现 MLP token 分块~~（已完成：32k 13.8GB、40k 15.1GB）。
2. 思考 KV 预留改为按实际最大 prompt+思考预算。
3. ~~评估 KV 格式（int8 / FP8 / MXFP4）~~（已完成：FP8 最优，省 0.6GB，66.4%）。
4. `o` 先改 bf16；`h` 视质量再定。
5. 完成 OpenRouter 细分：小模型、决策/分类类、上下文与价格分布。
6. 若长上下文是产品目标：96GB PRO 做 256k；1M 放到 H200/B200 并验证 YaRN。
