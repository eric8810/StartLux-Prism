#!/usr/bin/env bash
# Build the CUTLASS SM120 MXFP4 launcher used by kernels/w4a4.py (STARTLUX_PRISM_CUTLASS=1).
# CUTLASS_HOME must point at CUTLASS 4.8.x (tested: 4.8.1). CUDA_ARCH defaults to sm_120a;
# RTX PRO Blackwell (sm_121) needs CUDA_ARCH=sm_121a.
set -euo pipefail
cd "$(dirname "$0")"
: "${CUTLASS_HOME:?set CUTLASS_HOME to the CUTLASS 4.8.x source tree}"
CUDA_ARCH="${CUDA_ARCH:-sm_120a}"
nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC -arch="$CUDA_ARCH" \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -I "$CUTLASS_HOME/include" -I "$CUTLASS_HOME/tools/util/include" \
  mxfp4_launcher_pp.cu -o libmxfp4_pp.so
echo "wrote $(pwd)/libmxfp4_pp.so ($CUDA_ARCH)"
