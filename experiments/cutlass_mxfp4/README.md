# CUTLASS SM120 MXFP4 experiment (2026-10-25)

Standalone: `nvcc` build of CUTLASS 4.8.1's SM120 block-scaled GEMM, changed from the
79c example (MXFP8/MXFP6) to **MXFP4 x MXFP4** (`mx_float4_t<float_e2m1_t>`, scale ue8m0,
vector 32), bf16 output, 128x128x128 tile, `KernelScheduleAuto` (cooperative), TN.

## Build

```bash
CUTLASS=/home/eric8810/code/startlux/.deps/cutlass   # v4.8.1, outside the repo
nvcc -std=c++17 -O3 -arch=sm_120a --expt-relaxed-constexpr --expt-extended-lambda \
  -I $CUTLASS/include -I $CUTLASS/tools/util/include -I . \
  mxfp4_gemm.cu -o mxfp4_gemm
```

`mxfp4_gemm --m=1024 --n=24576 --k=4096 --iterations=10` checks the kernel against CUTLASS's
host reference (`GettBlockScaling`, random physical operands) and prints ms / GFLOPS.

## Results (RTX 5060 Ti, idle GPU beside the 32k server)

Verified (Disposition: Passed) at every shape below.

| shape (N,K) | M | CUTLASS | Triton (shape table) | speedup |
|---|---|---|---|---|
| gate\|up 24576x4096 | 1 | 0.137 ms | 0.212 ms | 1.55x |
| | 64 | 0.142 | 0.226 | 1.59x |
| | 1024 | 0.642 | 0.969 | 1.51x |
| | 4096 | 2.486 | 3.662 | 1.47x |
| down 4096x12288 | 1 | 0.0447 | 0.111 | 2.48x |
| | 64 | 0.0467 | 0.132 | 2.83x |
| | 1024 | 0.342 | 0.571 | 1.67x |
| | 4096 | 1.186 | 2.094 | 1.77x |

Caveats:

- `Avg runtime` includes `gemm.initialize` per iteration and the same weights are reused across
  iterations (L2 warm); Triton's numbers are `do_bench` (L2 flushed). The engine's numbers will
  differ, especially at M=1 (down's 25 MB fits L2).
- CUTLASS's scale layout is the interleaved `Sm1xxBlkScaledConfig`; our weights are row-major
  `[N, K/32]`. The bridge (load-time repack, one copy) and the fp32 residual epilogue are not
  done yet.
- The output is not compared with Triton yet; only with CUTLASS's own reference. Bit-exactness
  across the two is not expected (tile/K order differ).

## Estimate

MLP GEMM is ~40-50% of prefill: a 1.5-1.7x GEMM gives ~1.15-1.25x end-to-end prefill.
Decode is bandwidth-bound but CUTLASS's TMA/prefetch also wins there on small M; the engine
A/B has to measure it.

## Next

1. Layout bridge: fill CUTLASS SFA/SFB from row-major scales, A/B bytes from our packed nibbles;
   compare output with `w4a4.gemm` on the same logical data.
2. fp32 residual epilogue (`epi=1`).
3. Python wrapper behind `w4a4.gemm` (SM120 only), Triton fallback.
4. Engine A/B: 217 items, thought behavior, prefill/decode latency.
5. SM100/SM103 builders written but not validated (no hardware).

## SM100 / SM103 compile check

- **SM100**: the same template with `ArchTag = cutlass::arch::Sm100` and `-arch=sm_100a` compiles
  (`mxfp4_gemm_sm100.cu`), so `mx_float4_t` goes through the SM100 block-scaled UMMA builder. Not run
  (no SM100 hardware here).
- **SM103**: the naive template does **not** build (`Could not build a collective for given parameters`).
  The FP4 Ultra path needs example 89's structure: `MmaTileShape1Sm = 128x256x768`,
  `MmaTileShape2Sm = 256x256x768`, dynamic cluster (1 or 2), epilogue `NoSmemWarpSpecialized{1,2}Sm`,
  schedule `KernelTmaWarpSpecialized{1,2}SmBlockScaledMxNvf4UltraVs16Sm103`, and A/B as
  `cute::tuple<ElementA, ElementSFA>`. That is a separate instantiation, not an arch-tag switch.

## Layout bridge: bit-identical with Triton

The logical fill (A(m,k), B(n,k) as e2m1 codes; SFA(m,k), SFB(n,k) as ue8m0 exponents)
was checked against the Triton kernel on the same codes/scales:

| case | shape | Triton vs CUTLASS |
|---|---|---|
| epi 0 (bf16 out) | M=128, N=24576, K=4096 | **max abs 0, exact 1.0** |
| epi 1 (fp32 residual) | M=128, N=4096, K=12288 | **max abs 0, exact 1.0** |

So the data path does not change decisions: the nibble order, scale interpretation, K order
and epilogue match Triton. `mxfp4_gemm_epi1.cu` is the fp32 C/D instantiation (alpha=beta=1).

## Remaining for engine integration

1. GPU repack: B must be N-contiguous (CUTLASS `ColumnMajor`, `UMMA::Major::K`); our weights are
   K-contiguous `[N, K/2]`. Scales go to the interleaved `Sm1xxBlkScaledConfig` layout. One copy
   only: repack at load and free the row-major weights on SM120.
2. A data can stay as-is (row-major, K-contiguous); A scales are repacked per call (M x K/32).
3. A small C ABI launcher (`.so`, ctypes) or the existing torch extension; then
   `w4a4.gemm` picks CUTLASS on sm_120/121 and Triton otherwise.
4. Engine A/B: 217 items, thought behavior, prefill/decode latency; the kernel is bit-identical,
   so the gold/mean |dp| should not move.

## Engine integration (SM120, opt-in)

`mxfp4_launcher_pp.cu` is the launcher the engine calls: cooperative **Pingpong** schedule,
128x128x128, A/B data used as-is (both are K-contiguous and the nibble order matches), weight
scales repacked once (`cutlass_prepare_sfb`) and activation scales per call (`cutlass_prepare_sfa`).
No weight duplicate: only the interleaved scale buffer (3.1 MB for gate|up, 1.6 MB for down) is added.

```bash
CUTLASS_HOME=/path/to/cutlass ./build.sh
STARTLUX_PRISM_CUTLASS=1 STARTLUX_PRISM_CUTLASS_LIB=$PWD/libmxfp4_pp.so \
  STARTLUX_PRISM_MXFP4_MLP=$M/mxfp4/mlp_smooth.pt python -m startlux_prism.server ...
```

Measured on the RTX 5060 Ti, 32k, FP8 KV, think_seqs=4, smoothed MLP, same binary/config:

| | Triton | CUTLASS | |
|---|---|---|---|
| 32 read-out items, mean latency | 104.8 ms | **89.9 ms** | **1.17x** |
| 1,799-token read-out, p50 | 222.1 ms | **192.6 ms** | **1.15x** |
| read-out C=1 | 9.1 req/s | **10.6 req/s** | 1.16x |
| read-out C=64 | 9.0 req/s | **9.6 req/s** | 1.07x |
| 8 thoughts, C=4, budget 512 | 27.6 s | **24.3 s** | 1.14x |
| GPU memory at 32k | 13.6 GB | 14.1 GB | +0.5 GB |

The 32-item A/B gives **32/32 the same choice and max |dp| = 0** (bit-identical), because the
kernel matches Triton per GEMM (epi 0 and 1). The thought decoder still uses the int8 MLP; only
prefill changes.

Open items:

- The `.so` is not committed; build it with `build.sh` (CUTLASS 4.8.x, CUDA 12.8+).
- SM121 (RTX PRO Blackwell) needs `CUDA_ARCH=sm_121a`; not tested.
- SM100 compiles with the same builder (`mxfp4_gemm_sm100.cu`); SM103 needs the FP4 Ultra
  structure (example 89). Neither was run (no hardware).
