# Blackwell 与显存优化：整体结论（2026-10-25）

## 0. 一句话

在 RTX 5060 Ti 16GB 上，**原生 MXFP4 只放在 MLP**、attention/GDN/embedding 保持 int8、KV 用 FP8，
是当前唯一同时满足「质量、显存、速度」的配置：

```
--precision mxfp4 --calib int8_calib.pt --kv-fp8 --max-tokens 32768 --think-seqs 4
STARTLUX_PRISM_MXFP4_MLP=$M/mxfp4/mlp_smooth.pt
STARTLUX_PRISM_MLP_CHUNK=4096          # 默认
```

- **8.0GB**（1k，无思考），**66.8%**（217 条，与 int8 同分）
- **1k 单请求 146.5ms**，比 int8 的 183.4ms 快 **1.25x**
- **32k 上下文 13.2GB**；32k + 4 条思考 13.9GB
- 单卡读入吞吐约 **9 req/s**；思考并发 4 条，约 **0.3 条/s**（budget 512）

## 1. Blackwell（sm_120）方向

### 1.1 原生 MXFP4 是核心收益

- `tl.dot_scaled` 在 sm_120 落到 `mma.sync ... mxf4nvf4.block_scale`，权重和激活都是 e2m1，
  每 32 个输入一个 e8m0 scale。
- 实测 FP4 GEMM **183–196 TF/s**（`tests/test_w4a4.py`），bf16 cuBLAS 是 48–51 TF/s，约 **4x**。
- 端到端 1k：**146.5ms vs int8 183.4ms（1.25x）**，vs int4 W4A8 231.6ms（1.58x）。
- 质量：平滑后的 MLP FP4 + FP4 decode 与 int8 **同分 66.8%**（217 条），没有掉点。

### 1.2 关闭的路径

| 路径 | 结果 | 原因 |
|---|---|---|
| 全 FP4 | 56.2%，126.2ms，约 6GB | 质量不可用；最快但只能作为上限参考 |
| W4A8-FP4（e2m1 权重 + e4m3 激活） | 40.8% vs FP4×FP4 60.2% | 共享量化网格被平滑抵消；层 0 相对误差 7.2% vs 0.9%；Triton 3.8 用 bf16 模拟 mxf8f6f4，慢 |
| C 变体（FP4 gu + int8 down） | 43.3% | 平滑已把 `1/s_down` 折进 up 行，数值不匹配；代码已删 |

PTX 层面 sm_120f 支持 e2m1 block-scaled MMA；瓶颈是 **Triton 3.8 没有原生 lowering**，不是硬件。
要再快只能写 CUTLASS/CUDA 的 SM120 原生 kernel（同样的数学，质量不会变）。

### 1.3 今天的增量

**形状表**（照上游 W4A8 的做法）：

- `bench/w4a4_table.py` + `kernels/w4a4_table_rtx5060ti.json`：gate|up `24576×4096`、down `4096×12288`，
  1–8192 行、epi 0/1，14 个 tile 配置。
- 单 GEMM **1.03–1.19x**；端到端 1,798 token prefill **234.0 → 225.4ms（1.038x）**。
- BK 固定 128，配置只改 BM/BN，输出**逐位一致**；`tests/test_w4a4.py` 的 batch invariance 仍通过。

**上游 Hopper W4A8**：

- 已合并（`bed4529`）：H200 上 406–430 TOPS vs 201–271，端到端 1.16–1.55x。
- 只影响 sm_90，不影响 5060 Ti；借鉴的是它的「逐位一致 + 按形状表 + 重测」方法。

**CUTLASS SM120 集成（今天新增，opt-in）**：

- MLP 的 MXFP4 GEMM 换成 CUTLASS 4.8.1 SM120 block-scaled kernel（`mx_float4_t`，Pingpong）；
  A/B 数据直接复用，只重排 scale，无第二份权重。
- 端到端 32 条：**32/32 同选择，max |Δp| = 0**；平均 **89.9 vs 104.8 ms（1.17x）**；
  1,799 token **192.6 vs 222.1 ms（1.15x）**；C=1 **10.6 vs 9.1 req/s**；思考 8 条 **24.3 vs 27.6 s**。
- 32k 显存 14.1 vs 13.6GB。仅 sm_120/121；`.so` 需用 `experiments/cutlass_mxfp4/build.sh` 构建。
- SM100 同一模板编译通过（未运行）；SM103 需要 FP4 Ultra 的独立 1Sm/2Sm 结构（未实现）。

**并发**（当前 32k、FP8 KV、think_seqs=4，64 条混合长度）：

| 并发 C | 吞吐 | p50 | p95 |
|---|---|---|---|
| 1 | 9.1 req/s | 87 ms | 243 ms |
| 8 | 8.7 req/s | 868 ms | 1.03 s |
| 32 | 8.9 req/s | 2.02 s | 3.75 s |
| 64 | 9.0 req/s | 3.84 s | 7.11 s |

- 吞吐**不随并发增长**：prefill 已经把单卡算力吃满（约 8k token/s，平均每条 ~880 token）。
  加并发只是排队。p50 < 1s 约 8 条，< 2s 约 32 条，< 4s 约 64 条。
- **思考**：`think_seqs=4` 最多 4 条同时 decode；8 条 C=4、budget 512 总耗时 27.6s（think_seqs=1 是 69.2s，
  **2.5x**），吞吐约 0.3 条/s；C=8 时后 4 条排队，总耗时翻倍。
- **长上下文**：32k 基本一次一条，17.4k prefill 2.8s，32k 约 5.5s。
- 想要更高读入吞吐只能 **DP**（多卡各跑一份）；TP 降单条延迟，且当前只支持 int8，MXFP4 用不了。

## 2. 显存方向

### 2.1 从两份权重到单份：13.6GB → 8.0GB

之前 int8 模型保留 int8 MLP 副本给思考解码，启动 13.6GB。改成**单一权重集**：

- attention / GDN / embedding / thought head：int8
- MLP：MXFP4（prefill 和思考 decode 共用）
- **8.0GB**，省 **5.6GB**，质量 66.8%，1k 146.5ms。

| 配置 | 显存 | 质量 | 1k |
|---|---|---|---|
| int8（单份） | 11.3 GB | 66.8% | 183.4 ms |
| int4 W4A8 | 8.0 GB | 64.1% | 231.6 ms |
| **MLP FP4（平滑，单份）** | **8.0 GB** | **66.8%** | **146.5 ms** |
| 全 FP4 | ~6 GB | 56.2% | 126.2 ms |

### 2.2 MLP 分块：32k 从 OOM 到可用

- MLP 是逐 token 的，`[T, 2*ffn]` 中间量不需要跨整个 batch；按 `STARTLUX_PRISM_MLP_CHUNK=4096`
  分块，缓冲按 `min(T, chunk)` 分配。
- 32k：OOM → **13.8GB**（bf16 KV）；40k：**15.1GB**。
- 延迟基本不变（17.8k：2.91s vs 分块前 2.88s）。

### 2.3 KV 格式：FP8 是推荐

统一 2048 思考 cap、217 条：

| KV | gold | 思考条数 | 思考 token | 32k 显存 |
|---|---|---|---|---|
| bf16 | 66.8% | 76 | 52,030 | 13.8 GB |
| **FP8 e4m3** | **66.4%** | 76 | 53,303 | **13.2 GB** |
| int8 | 65.0% | — | — | — |
| MXFP4 | 63.1% | 62 | 42,129 | 12.9 GB |

- FP8 几乎无损、思考行为与 bf16 一致，省 0.6GB → **推荐**。
- int8 没有优势；MXFP4 改变思考行为，不建议用于决策。
- 早先 int8「思考 3 倍」是 4096 cap 造成的，不是格式。

平均概率差（相对 int8，217 条）：

| 配置 | mean Δp | 说明 |
|---|---|---|
| bf16 KV | 6.41% | 分数 66.8%，与 int8 同分 |
| FP8 KV | 6.53% | 推荐 |
| MXFP4 KV | 7.30% | 最大 |
| `o` bf16 | 4.95% | 已回退 |

`o`（GDN 输出）改 bf16 只省 0.27GB，组合评估降到 62.7%，mean Δp 4.95% → 回退，保持 fp32。

### 2.4 当前显存账

- 固定部分约 **7.9–8.1GB**：权重 6.4GB（FP4 MLP 2.4 + int8 attention/GDN 2.0 + embedding 1.0 +
  thought head 1.0）+ CUDA context / allocator / graph pool 约 1.5GB。
- 增长约 **256KB/token**；其中 KV bf16 32KB/token、FP8 16KB/token，其余是激活/scratch。
- **32k 13.2GB**（FP8）；32k + 4 条思考 **13.9GB**；40k 15.1GB（bf16 KV）。
- 16GB 卡上 **32k 是实用上限**，28k 只剩 0.4GB。

## 3. 今天实际改了什么

1. **合并上游** `d29cd2e`（Instant 双向 attention + Hopper W4A8），解 `attn.py` / `model.py` 冲突，
   把三种 KV 模式和 `BIDIR` 合到同一个 kernel。
2. **验证**：`tests/test_w4a4.py`（batch invariance）、`tests/test_bidir.py`（双向/causal/int8 scores）、
   真实服务读入 + 思考（FP8 KV）。
3. **FP4 形状表**：`bench/w4a4_table.py`、`w4a4_table_rtx5060ti.json`、`pick()` 查表。
4. **mean |Δp| 指标**：写进 `docs/context-memory.md`，比只看 gold 更能暴露量化差异。
5. **并发实测**：读入 9 req/s、think_seqs 4 vs 1、int8 vs FP4。
6. **提交**：`bed4529`（合并）、`33ad619`（形状表 + 文档），已推到 fork `main`
   https://github.com/eric8810/StartLux-Prism

## 4. 结论

- Blackwell 上 **原生 FP4 是真实收益**（GEMM 4x bf16，端到端 1.25x int8），但只适合放在 MLP；
  全 FP4 和 W4A8-FP4 都已关闭。
- 显存的主要杠杆是 **单份权重**（-5.6GB）和 **MLP 分块**（32k 从 OOM 到 13.8GB），KV 只占增长的一小部分；
  FP8 KV 再省 0.6GB。
- 单卡读入吞吐已经到顶（约 9 req/s），并发只增加排队；要提高吞吐用 DP；要提高单条速度需要 TP，但当前 TP 只支持 int8，MXFP4 用不了。
- SM120 的 CUTLASS 集成已验证：**1.15–1.17x 端到端，逐位一致**，是今天最实质的加速；
  下一步是把同一模板扩到 SM100（编译已验证）和 SM103（FP4 Ultra，未实现）。若要更大上下文，方向是
  **FP8 KV + 分块 + 更大显存的卡**（5090 32GB 约 190k，PRO 6000 96GB 约 735k）。
