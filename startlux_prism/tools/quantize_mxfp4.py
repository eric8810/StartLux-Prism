"""Pack a GGUF's linear projections to MXFP4 for the engine (`--precision mxfp4`).

    python -m startlux_prism.tools.quantize_mxfp4 MODEL.gguf OUT.pt [--mode max|mse] [--smooth CALIB.pt] [--alpha 0.5]

Groups: MLP (`gu` = gate|up, `down`), attention (`qkv`, `o`) and GDN (`inp` = qkv|z|beta|alpha, `out`).  Smoothing
(SmoothQuant style, `x/s`, `W*s`) folds into the norm weight for the input projections; for `o`/`out` the engine
scales the bf16 intermediate at runtime.  `down`'s input is `silu(gate)*up`, so `1/s_down` is folded into the `up`
rows of `gu`.

Streaming: one projection is dequantized, packed and dropped at a time; peak host memory is the GGUF plus one fp32
matrix.  The side file keys are `<layer>.<group>` and `<layer>.s_<group>` (the per-input-channel smoothing scales).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from startlux_prism import gguf_weights as GW          # noqa: E402
from startlux_prism.kernels import w4a4 as K          # noqa: E402


def smooth_scale(act, w, alpha):
    """Per-input-channel SmoothQuant scale, clamped like the int8 path."""
    wa = w.abs().amax(0).clamp_min(1e-5)
    return (act.clamp_min(1e-5) ** alpha / wa ** (1 - alpha)).clamp(1e-3, 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--mode", default="max", choices=("max", "mse"))
    ap.add_argument("--smooth", default="", help="int8 calibration file: SmoothQuant-style activation smoothing")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--groups", default="all", help="all, or a comma list: gu,down,qkv,o,inp,out")
    a = ap.parse_args()
    want = None if a.groups == "all" else set(a.groups.split(","))
    w = GW.load(a.gguf)
    calib = torch.load(a.smooth, map_location="cpu") if a.smooth else None
    blob = {}

    def take(name):
        return want is None or name in want

    def put(i, name, mat, s=None):
        p = K.pack(mat, a.mode)
        dq = K.dequant_packed(p.q, p.s)
        rel = (dq - mat).norm() / mat.norm()
        blob[f"{i}.{name}"] = (p.q, p.s)
        if s is not None:
            blob[f"{i}.s_{name}"] = s
        print(f"layer {i:2d}  {name:5s} rel {rel:.5f}", flush=True)

    for i, l in enumerate(w.layers):
        # ---- MLP
        if take("gu") or take("down"):
            gu = torch.cat([l.ffn_gate.dequant(torch.float32), l.ffn_up.dequant(torch.float32)], 0)
            dn = l.ffn_down.dequant(torch.float32)
            s_gu = None
            if calib is not None:
                ffn = dn.shape[1]
                s_gu = smooth_scale(calib[f"{i}:gu"], gu, a.alpha)
                s_dn = smooth_scale(calib[f"{i}:down"], dn, a.alpha)
                gu = gu * s_gu[None, :]
                gu[ffn:] = gu[ffn:] / s_dn[:, None]      # silu(gate) * (up / s_dn)
                dn = dn * s_dn[None, :]
            if take("gu"):
                put(i, "gu", gu, s_gu)
            if take("down"):
                put(i, "down", dn)
            del gu, dn

        # ---- attention or GDN projections
        if w.config.is_attention(i):
            if take("qkv"):
                qkv = torch.cat([l.wq.dequant(torch.float32), l.wk.dequant(torch.float32),
                                 l.wv.dequant(torch.float32)], 0)
                s = smooth_scale(calib[f"{i}:qkv"], qkv, a.alpha) if calib is not None else None
                if s is not None:
                    qkv = qkv * s[None, :]
                put(i, "qkv", qkv, s)
                del qkv
            if take("o"):
                o = l.wo.dequant(torch.float32)
                s = smooth_scale(calib[f"{i}:o"], o, a.alpha) if calib is not None else None
                if s is not None:
                    o = o * s[None, :]
                put(i, "o", o, s)
                del o
        else:
            if take("inp"):
                inp = torch.cat([l.wqkv.dequant(torch.float32), l.wz.dequant(torch.float32),
                                 l.wbeta.dequant(torch.float32), l.walpha.dequant(torch.float32)], 0)
                s = smooth_scale(calib[f"{i}:inp"], inp, a.alpha) if calib is not None else None
                if s is not None:
                    inp = inp * s[None, :]
                put(i, "inp", inp, s)
                del inp
            if take("out"):
                out = l.wout.dequant(torch.float32)
                s = smooth_scale(calib[f"{i}:out"], out, a.alpha) if calib is not None else None
                if s is not None:
                    out = out * s[None, :]
                put(i, "out", out, s)
                del out

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(blob, a.out)
    print("saved", a.out, f"{os.path.getsize(a.out) / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
