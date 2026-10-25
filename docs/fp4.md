# StartLux 9B：MXFP4 实现与评估记录

本文记录在本机（RTX 5060 Ti 16 GB）上把 StartLux-Decision v1.1 9B AutoThink 量化为 MXFP4 的实现、数据和结论。
量化对象与范围、代码位置、复现命令都在这里。

## 1. 目标

- **质量与 int4（Q4_K_M W4A8）接近**，不是对齐 int8。
- **尽可能全 FP4**（所有大线性投影），以吃到 Blackwell 的 FP4 张量核心。
- 速度、显存、质量三者一起记录，避免只谈其中一项。

非目标：NVFP4、稀疏（2:4）、CUTLASS 替换（见第 9 节，后续单独评估）。

## 2. 环境

| 项 | 值 |
|---|---|
| GPU | RTX 5060 Ti 16 GB（sm_120，15.5 GiB 可用） |
| 驱动 | 595.84，CUDA 13.2 |
| 工具链 | torch 2.14.1+cu130，Triton 3.8.0，nvcc 13.3 |
| 模型 | `StartLux-Decision-v1.1-9B-AutoThink`（底座 Qwen3.5-9B，32 层，3 GDN : 1 全注意力） |
| 权重来源 | Q4_K_M（W4A8 基线）、Q8_0（int8 基线与 FP4 量化源） |

## 3. 格式

**MXFP4**：

- 数值 e2m1：`0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6`。
- 每 32 个连续输入一个 e8m0 scale（2 的幂，存储 `e + 127` 的 uint8）。
- 权重：`q` uint8 `[N, K/2]`（k 偶数在低 4 位）+ `s` uint8 `[N, K/32]`。
- 激活：同一布局，按 token 动态量化。
- GEMM：`tl.dot_scaled` → `mma...kind::mxf4nvf4.block_scale.scale_vec::2X...ue8m0`，fp32 累加。

scale 选择：`max` 或 `mse`（在相邻的两个 2 的幂里取该 32 元素块 MSE 更小的一个）。

## 4. 激活平滑（SmoothQuant 风格）

对每个投影组，逐输入通道：

```
x' = x / s        W' = W * s        x' W'ᵀ = x Wᵀ
s = clamp(amax_act^alpha / amax_w^(1-alpha), 1e-3, 1e3)
```

`amax_act` 来自 int8 校准文件（`int8_calib.pt`，128 组：每层 `gu/down/qkv/o/inp/out`）。

折叠方式（运行时零额外算子）：

| 投影 | 做法 |
|---|---|
| `gu`（gate+up） | `post_norm` 改为 `post_norm / s_gu`；权重乘 `s_gu` |
| `down` | `down` 的输入是 `silu(gate) * up`；把 `1/s_down` 折进 `up` 的输出通道，再把 `w_down * s_down` |
| `qkv` / `inp` | `attn_norm` 改为 `attn_norm / s`；权重乘 `s` |
| `o` / `out` | 权重乘 `s`；运行时把中间结果（attention mix / GDN 输出）除以 `s` 再量化 |

## 5. 层范围

| 模块 | 当前 | 计划 |
|---|---|---|
| MLP `ffn_gate/up/down` | **FP4 + 平滑** | 保持 |
| 全注意力 `wq/wk/wv/wo` | int8 | **FP4 + 平滑** |
| GDN `wqkv/wz/wout` | int8 | **FP4 + 平滑** |
| GDN `wbeta/walpha`、`conv1d` | 高精度 | 保持 |
| norm / embedding | int8 / bf16 | 保持 |
| `lm_head` | 只读 26 行，fp32 | 保持 |
| 思考解码 | int8 MLP 副本 | 后续实现 FP4 decode |

## 6. 代码

| 文件 | 内容 |
|---|---|
| `startlux_prism/kernels/w4a4.py` | MX4 打包、激活量化、`tl.dot_scaled` GEMM、max/MSE scale |
| `startlux_prism/tools/quantize_mxfp4.py` | Q8_0 GGUF → `mxfp4/mlp*.pt`，流式；`--smooth` / `--alpha` |
| `startlux_prism/tools/make_int8_calib.py` | CPU fp32 参考生成 `--calib`（W8A8 与平滑共用） |
| `startlux_prism/engine/model.py` | `--precision mxfp4`、`_mxfp4_weights/_mxfp4_mlp_weights`、`_body_mxfp4*`、`_head_mxfp4*`、`post_norm_fp4` |
| `startlux_prism/server.py` | `--precision mxfp4` |
| `tests/test_w4a4.py` | 打包/反量化、激活、GEMM 对拍、batch invariance、吞吐 |

## 7. 实测数据

### 7.1 延迟（p50，单请求，think_mode=never）

| 场景 | int8 | int4（W4A8） | FP4（MLP-only） |
|---|---|---|---|
| 1 题 256 tok | 52.9 | 71.4 | 49.1 |
| 1 题 1k tok | 183.4 | 231.6 | 161.5 |
| 1 题 3k tok | 519.5 | 657.7 | 503.1 |
| 3 题 1k tok | 189.3 | 238.7 | 165.5 |

README 单请求：int8 38.6，int4 56.8，FP4 39.2 ms。

FP4 相对 int4 约 **1.4–1.6x**（1k：147–161 vs 232 ms）。

### 7.2 质量（DecisionPivot 13 family × 18 条，217 条有效，think_mode=auto，gate 0.6）

| 方案 | gold accuracy | 相对 int8 | argmax 一致 | 高置信一致 | mean \|Δp\| | p95 |
|---|---|---|---|---|---|---|
| int8 | 66.8% | — | — | — | — | — |
| int4（Q4_K_M W4A8） | 64.1% | -2.8 | — | — | — | — |
| FP4 max | 62.7% | -4.1 | 84.8% | 90.4% | 7.83% | 39.7% |
| FP4 MSE | 61.8% | -5.1 | 84.8% | 88.1% | 8.37% | 43.0% |
| **FP4 平滑 0.5** | **65.0%** | **-1.8** | 83.4% | 89.8% | 8.05% | 42.2% |
| FP4 平滑 0.75 | 64.1% | -2.8 | 86.2% | 92.7% | 6.82% | 28.5% |

结论：

- MLP-only + 平滑 **已经达到或略好于 int4**（65.0 vs 64.1），速度快 1.4–1.6x。
- 高置信子集接近 int8（-0.5 分）。
- 217 条上 1 分 ≈ 2 条题，噪声约 ±3 分；需要扩到 500 条才能区分 0.5 / 0.75。

### 7.3 CPU 模拟（MLP-only，6 条）

| 方案 | mean \|Δp\| | p95 |
|---|---|---|
| MXFP4 | 6.45% | 17.2% |
| NVFP4 | 6.15% | 13.6% |
| MXFP4 + 平滑 | 3.95% | 7.1% |

NVFP4 只比 MXFP4 好一点，说明瓶颈是 4-bit 激活精度/离群值，不是 scale 粒度。

### 7.4 显存

| 方案 | 权重 | 显存 |
|---|---|---|
| int8 | 9.5 GB | 11.3 GB |
| int4 | 5.6 GB | 10.3 GB |
| FP4（MLP-only，含思考 int8 副本） | 12 GB | 12.6 GB |

当前 FP4 显存反而最大，因为思考解码保留了 int8 MLP 副本（约 4.8 GB）。实现 FP4 decode 后才能降下来。

## 8. 已知问题

1. **统计功效不足**：217 条，±3 分。需要 500–600 条。
2. **显存**：int8 副本未去掉；全 FP4 + int8 decode 预计约 13–14.5 GB，接近上限。
3. **思考路径**：仍用 int8 MLP；最终答案按 0.25 FP4 / 0.75 int8 混合，低置信题仍受 FP4 影响。
4. **batch invariance**：int8 的位级一致在 FP4 上不成立（fp32 累加）。当前 Triton 配置按形状固定 K 顺序，同形状内仍一致。
5. **int4 W4A4（int4 权重 + int4 激活）**：Prism 没有实现，也没有可用的 int4 MMA 引擎。若要对齐，需要额外实现或只做 CPU 模拟。

## 9. 后续

1. 把 attention / GDN 大投影接入 FP4（带平滑），跑 500 条验收（≥ int4 - 0.5）。
2. 实现 FP4 decode，去掉 int8 副本。
3. 评估 CUTLASS SM120 block-scaled GEMM（现成，有 `mxf4` 和 `nvf4`），目标 200 → 300+ TF/s；按 M 分派，小 M 保留 Triton。
4. 如果全 FP4 掉分超过预算，再考虑 AWQ/GPTQ / 旋转 / QAD。

## 10. 复现

```bash
# 1. int8 校准（CPU）
python -m startlux_prism.tools.make_int8_calib <Q8_0.gguf> <model_dir> <calib.pt> --threads 12

# 2. 量化 MLP（max / mse / 平滑）
python -m startlux_prism.tools.quantize_mxfp4 <Q8_0.gguf> <model_dir>/mxfp4/mlp_smooth.pt \
    --mode max --smooth <calib.pt> --alpha 0.5

# 3. 启动
STARTLUX_PRISM_MXFP4_MLP=<model_dir>/mxfp4/mlp_smooth.pt \
python -m startlux_prism.server --gguf <Q8_0.gguf> --model-dir <model_dir> \
    --precision mxfp4 --calib <calib.pt> --max-tokens 2048 --prefix-cache 2 \
    --prefix-cache-tokens 1024 --think-seqs 1 --port 8092

# 4. 单测
python tests/test_w4a4.py --quick
```

## 11. 全 FP4（2026-10-25）

开关：`STARTLUX_PRISM_FULL_FP4=1` + `STARTLUX_PRISM_MXFP4_MLP=<full file>`。

**范围**：MLP（gu/down）、全注意力（qkv/o）、GDN（inp/out）全部 MXFP4，全部带平滑；norm、conv、GDN 状态、
attention softmax/PV、embedding、lm_head 保持高精度。

**新增代码**：

| 位置 | 内容 |
|---|---|
| `quantize_mxfp4.py --groups all` | 打包 qkv/o/inp/out，写入 `s_qkv/s_o/s_inp/s_out` |
| `_add_full_fp4(w)` | 加载六组 FP4 权重，生成 `attn_norm_fp4`、`post_norm_fp4`、`o_inv`、`out_inv` |
| `_body_full_mxfp4` / `_head_full_mxfp4` | 全线性层 FP4 的 forward |
| `w4a4.quantize_act_scaled` | 逐通道缩放后量化（o/out 的平滑输入） |

**产物**：`mxfp4/full_smooth.pt`，3.68 GB。

**启动**：13.6 / 15.5 GiB（int8 副本仍保留给思考解码），加载 14.9 s。

**延迟（p50，think_mode=never）**：

| 场景 | int4 | int8 | FP4 MLP-only | **全 FP4** |
|---|---|---|---|---|
| 1 题 256 tok | 71.4 | 52.9 | 44.4 | **39.3** |
| 1 题 1k tok | 231.6 | 183.4 | 146.9 | **126.2** |
| 3 题 1k tok | 238.7 | 189.3 | 148.0 | **127.7** |

README 单请求：int4 56.8 → 全 FP4 28.3 ms（2.0x）；int8 38.6（1.36x）。

**质量**：待 217 条评估。冒烟请求出现了明显变化（team 从 billing 0.99 变成 shipping 0.73，none 0.23），
提示 attention/GDN 的 W4A4 损伤比 MLP 大，符合 NVIDIA 只把这两类放在 FP8 的选择。

### 11.1 全 FP4 的两个 bug（已修）

第一次全 FP4 评估得到 -19.8 分，是 bug，不是量化本身：

1. **GDN 输出少了 `ssm_norm`**：int8/W4A8 路径在 `out` 投影前会跑 `KQ.gated_norm_g`；全 FP4 路径直接量化
   `b["o"]`，漏了这步。已改为 `KG.gated_norm(..., mixg, ...)`。
2. **输出 stride 错位**：attention 的 qkv 投影宽 10240，但写进了行 stride 为 12352 的 `proj` buffer；
   FP4 GEMM 原来按 `N` 当 stride 写，数据错位。已给 kernel 加 `OUTS` 参数，调用处传 `proj.stride(0)`。

修复后 README 请求恢复为 billing 0.988，延迟 22.7 ms。

**修复后延迟（p50，think_mode=never）**：

| 场景 | int4 | int8 | FP4 MLP-only | 全 FP4 |
|---|---|---|---|---|
| 1 题 256 tok | 71.4 | 52.9 | 44.4 | **40.0** |
| 1 题 1k tok | 231.6 | 183.4 | 146.9 | **129.2** |

全 FP4 相对 int4 **1.79x**，相对 int8 **1.42x**。

质量复测进行中（217 条）。

### 11.2 修复后的全 FP4 质量（217 条）

| 方案 | gold accuracy | 相对 int8 | 相对 int4（W4A8） | argmax 一致 | 高置信一致 | mean \|Δp\| |
|---|---|---|---|---|---|---|
| int8 | 66.8% | — | +2.7 | — | — | — |
| int4（W4A8） | 64.1% | -2.8 | — | — | — | — |
| FP4 MLP-only + 平滑 0.5 | **65.0%** | -1.8 | **+0.9** | 83.4% | 89.8% | 8.05% |
| **全 FP4 + 平滑 0.5** | **56.2%** | **-10.6** | **-7.9** | 73.3% | 78.5% | 12.49% |

分家（gold accuracy）：

| family | int8 | int4 W4A8 | 全 FP4 |
|---|---|---|---|
| answerable_facts | 83.3 | 83.3 | 88.9 |
| perm_inherit | 44.4 | 50.0 | 55.6 |
| record_rule | 44.4 | 22.2 | 44.4 |
| toolcall_valid | 83.3 | 94.4 | 83.3 |
| clause_change | 77.8 | 66.7 | 61.1 |
| dep_resolve | 68.8 | 62.5 | 50.0 |
| filter_match | 88.9 | 77.8 | 55.6 |
| ledger_reconcile | 46.7 | 53.3 | 26.7 |
| policy_case | 77.8 | 66.7 | 55.6 |
| state_goal | 83.3 | 83.3 | 61.1 |
| sched_conflict | 56.2 | 50.0 | 43.8 |

结论：

- 纯 PTQ 的全 FP4（attention/GDN 也 W4A4）**比 W4A8 int4 低 7.9 分**，不能用。
- 损伤集中在需要精确比较/追踪的任务（filter_match、ledger_reconcile、state_goal、policy_case）。
- 与 NVIDIA 的选择一致：他们的 Qwen3.5 配方把 attention/GDN 放在 FP8，只把 MLP 放 FP4。
- MLP-only + 平滑是当前唯一达到 int4 质量的 FP4 方案（+0.9 分，相对 int4 1.57x）。
- 缺的对照：**int4 W4A4**（int4 权重 + int4 激活）没有现成引擎，需要模拟或实现，才能判断全 FP4 是否只是“和 W4A4 int4 同级”。

## 12. 单一权重集的混合精度（8.0GB）

问题：MLP-only 配置里 int8 MLP 和 FP4 MLP 同时常驻（4.8 + 2.4GB），只为让思考解码读 int8；
全 FP4 配置里 int8 全模型（9.5GB）与 FP4 全模型（3.7GB）同时常驻。两种都是双份，没有价值。

改法：每组只存一份，按组选精度。

| 组 | 格式 | 显存 |
|---|---|---|
| MLP（gu/down） | MXFP4 | 2.4 GB |
| attention（qkv/o） | int8 | 0.34 GB |
| GDN（inp/out） | int8 | 1.6 GB |
| embed | int8 | 1.0 GB |

- prefill：`_body_mxfp4_i8`（int8 attention/GDN + FP4 MLP），未变。
- decode：新增 `_layers_mxfp4`（int8 attention/GDN + FP4 MLP），思考不再读 int8 MLP。
- `_int8_weights(..., mlp=False)`：不再生成 int8 gu/down（省 4.8GB）。

结果（2048 token 配置）：

| 配置 | 加载后显存 |
|---|---|
| 旧的 MLP-only（双份 MLP） | 11.7 GB |
| **单份混合（MLP FP4 + attn/GDN int8）** | **8.0 GB** |
| 全 FP4 + int8 全模型 | 13.6 GB |

思考速度：44 token / 1.06 s ≈ 24 ms/token（int8 解码约 28 ms/token）。
质量复测进行中；prefill 算术未变，只有触发思考的样本可能变化。

### 12.1 思考解码改 FP4 的质量代价

同一 217 条、同一 prefill：

| 配置 | 显存 | gold | 相对 int8 | 相对 int4 |
|---|---|---|---|---|
| MLP FP4 + int8 decode（双份 MLP） | 11.7 GB | 65.0% | -1.8 | +0.9 |
| **MLP FP4 + FP4 decode（单份）** | **8.0 GB** | **62.2%** | -4.6 | **-1.9** |

思考是自回归的，解码端的任何舍入差异会被放大：两次运行 83.9% 的答案项相同，其余 16% 集中在触发思考的样本。
2.8 分接近这套 217 条的噪声底（约 ±3），要确认需要 500–600 条。

结论：单份权重把显存压到 8.0GB，但思考质量掉到 int4 以下。要同时满足“8GB + int4 质量”，
需要训练时方法（QAD）或只对部分层保留 int8 MLP。

## 13. FP4 权重 × FP8 激活（mxf8f6f4）：硬件支持，Triton 不支持

思路：权重保持 MXFP4（0.5 字节/参数），激活用 FP8 e4m3，一个 `mma.kind::mxf8f6f4` 完成。

**结论：sm_120f 硬件支持，CUTLASS 已实现；Triton 3.8 后端没有 lowering，所以退化成 bf16 模拟。**

证据：

- PTX ISA：`mma.sync.aligned.m16n8k32.row.col.kind.block_scale.scale_vec::1X.f32.f8f6f4type.f8f6f4type.f32.ue8m0`，
  其中 `.f8f6f4type = {.e4m3, .e5m2, .e3m2, .e2m3, .e2m1}`；target note 明确列出
  `mma{.sp} with .e3m2, .e2m3, .e2m1 types and .kind, .block_scale` 支持 **sm_120f**。
- CUTLASS：`include/cute/arch/mma_sm120.hpp` 有
  `SM120_16x8x32_TN_VS<float_e4m3_t, float_e2m1_t, float, float_ue8m0_t, 32>`，PTX 正是
  `mma.sync...kind::mxf8f6f4.block_scale.scale_vec::1X...e4m3.e2m1...ue8m0`；
  另有 `sm120_blockscaled_mma_tma.hpp` collective，用 `float_e2m1_unpacksmem_t` 把全局的打包 4-bit
  在 TMA 载入 shared 时展开成 8-bit 容器，所以权重显存仍是 0.5 字节/参数。
- Triton 3.8：`tl.dot_scaled(..., "e4m3", ..., "e2m1")` 编译成功、数值也对（1e-6），但 PTX 是
  `mma.sync...bf16.bf16`——即 bf16 软件模拟。M=2048 时 10.3 ms vs 原生 FP4 2.1 ms（5 倍慢），
  M=1 时 0.436 vs 0.197 ms（2.2 倍慢）。

注意：mxf8f6f4 的 e2m1 操作数在寄存器/shared 中必须是 **8-bit 容器**（4-bit 值居中放在 bit 2-5），
所以要用 `ldmatrix...b8x16.b4x16_p64` 或等价方式从打包 4-bit 展开；不能直接把打包字节喂给 MMA。

### 13.1 这条路的代价和收益

`mxf8f6f4` 的形状是 m16n8k32，`mxf4nvf4` 是 m16n8k64，算力约为 FP4 的一半（FP8 级）。

| 方案 | 权重 | 激活 | 算力 | 1k prefill | 实测显存 | 质量 |
|---|---|---|---|---|---|---|
| 全 FP4 | e2m1 0.5B | e2m1 0.5B | 195 TFLOP/s | 129 ms | 约 6GB（单份） | 56.2% |
| **W4A8-FP4（mxf8f6f4）** | e2m1 0.5B | e4m3 1B | 约 100 TFLOP/s | 约 240 ms | 约 8GB | 待测，预计接近 int4 |
| W4A8-int4（Q4_K_M） | 4.5bit | int8 1B | 约 100 TOPS | 232 ms | **8.0GB** | 64.1% |
| int8 | int8 1B | int8 1B | 约 100 TOPS | 183 ms | 11.3GB | 66.8% |

注意：Q4_K_M 是 5.3GB 的 k-quant 文件（约 4.7 bit/权重，部分张量 Q6_K），int4 值在 kernel 里
展开成 int8 后走 int8 tensor core，**存储仍是 4-bit**，所以显存 8.0GB（含 2k 上下文和前缀缓存）。

所以 W4A8-FP4 买到的是**显存**：全模型约 6GB，速度与 W4A8-int4 同级，不是全 FP4 的 1.8 倍。
要同时拿到 FP4 速度和 int4 质量，仍然只能靠训练时方法（QAD）。

工程成本：把 CUTLASS 的 SM120 blockscaled GEMM 接进 Prism（或自己写 inline PTX kernel），
主要难点是 scale-factor 的 CUTLASS 布局和 TMA 描述符，预计数天。

### 13.2 用模拟 kernel 实测 W4A8-FP4 质量（未达预期）

在 Triton 的 bf16 模拟下（数值已验证：M=1024 时比 FP4×FP4 更接近精确 FP4 权重模型 168 倍），
同一 217 条集、`think_mode=never`、前 200 条：

| 配置 | gold（191 条有效） | 平均置信度 |
|---|---|---|
| FP4×FP4（smooth 0.5） | **60.2%** | 0.689 |
| W4A8-FP4（smooth 0.5） | **40.8%** | 0.802 |

act8 更自信但更差，且分家差异最大的是 toolcall_valid（18/18 → 6/18）。这不是量化噪声，
更像是数据流里还有一处没对上；逐层 dump 显示第 0 层 hidden 差 6.7%，到第 30 层完全分叉
（FP4 的 residual 涨到 178，act8 只有 52）。

在这条异常查清之前，不应投入 CUTLASS 实现原生 mxf8f6f4：原生 kernel 的数学与模拟一致，
模拟结果差 19 分，说明问题不在指令，而在当前接线或平滑约定。

### 13.3 查清了：不是接线 bug，是 FP4 激活与 FP4 权重配套抵消

用 int8 引擎导出同一请求的第 0 层 hidden state 作为参照：

| 路径 | 第 0 层 h 与 int8 的相对差 |
|---|---|
| FP4×FP4（平滑 0.5） | **0.9%** |
| W4A8-FP4（平滑 0.5） | 7.2% |

也就是说，FP4×FP4 几乎精确复现了 int8 的轨迹；把激活换成 e4m3 反而偏离 8 倍。

原因：FP4 权重（e2m1 + 每 32 个 e8m0）和 FP4 激活（e2m1 + 每 32 个 e8m0）使用**同一套量化网格和
同一个“amax/6 取 2 的幂”规则**，平滑参数又是两者一起调的。两边的舍入误差高度相关，乘积里互相抵消。
换成 e4m3（amax/448）后，网格和对齐方式变了，抵消关系消失。

四种组合（前 200 条、`think_mode=never`、191 条有效）：

| 权重/激活 | 平滑 | gold |
|---|---|---|
| FP4 / FP4 | 0.5 | **60.2%** |
| FP4 / FP4 | 无 | 50.3% |
| FP4 / FP8 | 0.5 | 40.8% |
| FP4 / FP8 | 无 | 44.0% |

结论：**W4A8-FP4 这条路关闭**。不是 Triton 模拟的问题，也不是 CUTLASS 能解决的——原生 mxf8f6f4
的数学与模拟一致，质量仍会在 41–44%。要让 FP8 激活配 FP4 权重达到 int4 质量，需要重新量化权重
（不同网格）或 QAT，不是换一条 MMA 指令能解决的。

## 14. Q4_K_M 的真实混合精度（实测 GGUF 张量类型）

| 张量 | 类型 |
|---|---|
| `ffn_down` | 16 层 Q6_K + 16 层 Q4_K |
| `attn_qkv`（GDN 输入） | 12 层 Q6_K + 12 层 Q4_K |
| `attn_v`（attention） | 4 层 Q6_K + 4 层 Q4_K |
| `output` | Q6_K |
| `ffn_gate/up`、`ssm_out`、`ssm_alpha/beta`、`attn_gate`、`token_embd` | Q4_K |

平均约 4.7 bit。敏感张量（`ffn_down`、部分 qkv/v）留在 6-bit，是 Q4_K_M 质量的关键。

对 FP4 的启示：`ffn_down` 敏感，可以只把 `ffn_gate/up` 放 FP4、`down` 留 W8A8（int8）。
int8 比 Q6_K 更准也更快，所以敏感层选 int8 而不是 Q6_K。

## 15. 修正 12.1：8GB 单份 + 平滑权重 = 66.8%

12.1 的 62.2% 用的是**未平滑** `mlp.pt`（启动脚本漏了 `STARTLUX_PRISM_MXFP4_MLP`）。
用 `mlp_smooth.pt`、单份权重、FP4 解码重测同一 217 条：

| 配置 | 显存 | gold | 相对 int8 | argmax 一致 |
|---|---|---|---|---|
| int8 | 11.3 GB | 66.8% | — | — |
| **MLP FP4 + FP4 decode（平滑）** | **8.0 GB** | **66.8%** | **±0** | 82.9% |
| MLP FP4 + int8 decode（平滑） | 11.7 GB | 65.0% | -1.8 | 83.4% |

单份权重 + FP4 解码在平滑权重下与 int8 打平，比 int8 快约 1.25 倍（147 vs 183 ms），省 3.3GB。
之前“FP4 解码掉 2.8 分”的结论是未平滑权重造成的假象，作废。

## 16. 最终结论（217 条 DecisionPivot，auto，17 个超长错误）

| 方案 | 显存 | gold | 1k 延迟 | 相对 int8 |
|---|---|---|---|---|
| int8 | 11.3 GB | 66.8% | 183 ms | — |
| int4 W4A8（Q4_K_M） | 8.0 GB | 64.1% | 232 ms | -2.8 |
| **MLP FP4 + FP4 decode（smooth 0.5，单份）** | **8.0 GB** | **66.8%** | **146.5 ms** | **0** |
| MLP FP4 + int8 decode（smooth 0.5） | 11.7 GB | 65.0% | 147 ms | -1.8 |
| 全 FP4 | 约 6 GB | 56.2% | 129 ms | -10.6 |
| W4A8-FP4（mxf8f6f4 模拟） | — | 41–44% | — | 关闭 |
| 只把 gu 放 FP4、down 留 int8 | 8.7 GB | 43.3% | — | 实现有双缩放错误，作废 |

推荐配置：

```
--precision mxfp4 --calib int8_calib.pt
STARTLUX_PRISM_MXFP4_MLP=$M/mxfp4/mlp_smooth.pt
```

单份权重（不再加载 int8 MLP），prefill 和思考解码都用 MXFP4 MLP，attention/GDN/embed 保持 int8。

相对 int4：同显存（8.0GB）、高 2.7 分、快 1.58 倍。
相对 int8：省 3.3GB、快 1.25 倍、这套 217 条上同分（145/217）。

注意：217 条的噪声约 ±3 分，B 与 int8 同分可能是巧合；要确认需 500–600 条。

## 17. 16GB 显存下的上下文上限（B 配置）

实测 `--max-tokens` 与启动后显存：

| max-tokens | 显存 | 结论 |
|---|---|---|
| 2,048 | 8.0 GB | 余量充足 |
| 4,096 | 8.7 GB | 余量充足 |
| 8,192 | 10.0 GB | 可用 |
| 16,384 | 12.1 GB | 推荐 |
| 24,576 | 14.1 GB | 实用上限（1.4GB 余量） |
| 28,672 | 15.1 GB | 极限，只剩 0.4GB |
| 32,768 | — | OOM |

长上下文延迟（think_mode=never）：

| 输入 | 延迟 | 吞吐 |
|---|---|---|
| 2.3k | 298 ms | 7,723 tok/s |
| 4.5k | 615 ms | 7,340 tok/s |
| 8.9k | 1,292 ms | 6,922 tok/s |
| 17.8k | 2,878 ms | 6,187 tok/s |

显存大头是激活缓冲区（`gu` 在 32k 时是 1.6GB），不是 KV。模型本身支持 262k 位置，但 16GB 卡上
`--max-tokens` 24k 是实用上限、16k 是稳妥值。当前服务按 24,576 启动。

## 18. FP4 GEMM 形状表（2026-10-25）

照上游 W4A8 的做法，给 MXFP4 GEMM 加了按 GPU、按形状的配置表和重测工具：

- `bench/w4a4_table.py`：对 MLP 的两个投影（gate|up `24576×4096`、down `4096×12288`）在 1–8,192 行、
  epi 0/1（bf16 输出 / fp32 残差加）上逐个测 tile 配置，最快者写入
  `startlux_prism/kernels/w4a4_table_<gpu>.json`（RTX 5060 Ti：`w4a4_table_rtx5060ti.json`）。
- `kernels/w4a4.py:pick` 按 `(N, K, epi)` 取不小于当前行数的最小条目；没有条目时回退到原规则。
  `STARTLUX_PRISM_W4A4_TABLE` 可指定别的表，`off` 关闭。
- BK 固定 128，BM/BN 只决定哪个 program 算哪个元素，不改变 K 的累加顺序；表的每个条目都和默认配置
  逐位比较（`--verify`），`tests/test_w4a4.py` 的 batch invariance 也仍然通过。

RTX 5060 Ti 的收益（同一形状、同一行数，表 vs 原规则）：

| 形状 | 行数 | 表 | 时间 | 规则 | 提升 |
|---|---|---|---|---|---|
| gate\|up | 1 | (64,32,8,4,3) | 0.21 ms | 0.22 ms | 1.04x |
| gate\|up | 4,096 | (256,128,8,8,3) | 3.67 ms | 4.12 ms | 1.12x |
| down | 1 | (32,128,8,4,3) | 0.11 ms | 0.13 ms | 1.19x |
| down | 128 | (128,128,8,8,3) | 0.15 ms | 0.18 ms | 1.18x |
| down | 4,096 | (256,128,8,8,3) | 1.99 ms | 2.07 ms | 1.04x |

端到端（1,798 token 的随机 prompt，think_mode=never，同一二进制）：

| 表 | 延迟（5 次中位数） |
|---|---|
| off | 234.0 ms |
| **on** | **225.4 ms** |

约 **1.038x**。收益不大，因为 MLP 只占 prefill 的一部分；表的价值是免去手工猜 tile，并在换卡时用
`python bench/w4a4_table.py` 重测（`--recheck` 只对比表中条目与原规则）。

## 19. CUTLASS SM120 引擎集成（2026-10-25）

在 SM120（RTX 5060 Ti）上把 MLP 的 MXFP4 GEMM 换成 CUTLASS 4.8.1 的 SM120 block-scaled
kernel（`mx_float4_t`，ue8m0，SF=32，128×128×128，Pingpong），opt-in：

```
STARTLUX_PRISM_CUTLASS=1
STARTLUX_PRISM_CUTLASS_LIB=<repo>/experiments/cutlass_mxfp4/libmxfp4_pp.so
```

数据路径：

- 权重和激活的 **数据字节直接复用**（两者的 K 都是连续的，nibble 顺序一致）；
- 权重 scale 在加载时重排一次（`cutlass_prepare_sfb`），激活 scale 每次调用重排（`cutlass_prepare_sfa`）；
- 不产生第二份 2.4GB 权重；只多 3.1MB（gate|up）+ 1.6MB（down）每层的交错 scale。

同二进制、同配置（32k、FP8 KV、think_seqs=4、平滑 MLP）的 A/B：

| | Triton | CUTLASS | |
|---|---|---|---|
| 32 条读入，平均延迟 | 104.8 ms | **89.9 ms** | **1.17x** |
| 1,799 token 读入，p50 | 222.1 ms | **192.6 ms** | **1.15x** |
| 读入 C=1 | 9.1 req/s | **10.6 req/s** | 1.16x |
| 读入 C=64 | 9.0 req/s | **9.6 req/s** | 1.07x |
| 思考 8 条 C=4 budget 512 | 27.6 s | **24.3 s** | 1.14x |
| 32k 显存 | 13.6 GB | 14.1 GB | +0.5 GB |

**32 条结果 32/32 同选择，max |Δp| = 0**（逐位一致）：kernel 在 epi 0/1 上都与 Triton 的
GEMM 一致，引擎的思考解码仍走 int8 MLP，只有 prefill 变化。

限制：

- 仅 sm_120/121；`.so` 不入库，用 `experiments/cutlass_mxfp4/build.sh` 构建（CUTLASS 4.8.x，
  CUDA 12.8+）。
- SM100 的同一模板可编译（`mxfp4_gemm_sm100.cu`），未在硬件上运行；SM103 需要示例 89 的
  FP4 Ultra 1Sm/2Sm 结构，未实现。
- 默认关闭，避免在没有构建 `.so` 的机器上启动失败。
