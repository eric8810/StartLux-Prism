"""Create the int8 activation-range file Prism's W8A8 path reads (--calib) on CPU.

    python -m startlux_prism.tools.make_int8_calib MODEL.gguf MODEL_DIR CALIB.pt [--threads 12]

The engine's own calibration runs a bf16 forward, which needs the 18.7 GB bf16 copy resident: impossible on a 16 GB
GPU.  This does the same recording on the fp32 reference (Q8_0 kept as int8, one matrix dequantized at a time), so the
engine can start with --calib and build its int8 weights straight from Q8_0.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from startlux_prism import gguf_weights, jevfmt as J, reference as REF   # noqa: E402
from startlux_prism.calibration import CALIBRATION                      # noqa: E402
from startlux_prism.prompt import Tokenizer, rows_for                   # noqa: E402


def groups(w):
    """id(Q8 matrix) -> ((layer, name), matrices) - the engine's calibration keys (calib records (i, name))."""
    out = {}
    for i, l in enumerate(w.layers):
        sets = [("gu", (l.ffn_gate, l.ffn_up)), ("down", (l.ffn_down,))]
        if w.config.is_attention(i):
            sets += [("qkv", (l.wq, l.wk, l.wv)), ("o", (l.wo,))]
        else:
            sets += [("inp", (l.wqkv, l.wz, l.wbeta, l.walpha)), ("out", (l.wout,))]
        for name, mats in sets:
            for m in mats:
                out[id(m)] = ((i, name), mats)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = Tokenizer(a.model_dir)
    ref = REF.Reference(gguf_weights.load(a.gguf), device="cpu")
    ref.set_letters(tok.letters)
    G = groups(ref.w)
    exact = REF.linear
    amax = {}

    def recording(x, q8):
        key, _ = G[id(q8)]
        m = x.abs().amax(0)
        amax[key] = torch.maximum(amax[key], m) if key in amax else m
        return exact(x, q8)

    REF.linear = recording
    for n, (state, questions) in enumerate(CALIBRATION):
        recs = [J.from_systemone(state, q) for q in questions.values()]
        for r in rows_for(tok, recs, list(questions)):
            ref.forward(r.ids)
        print(f"calibration request {n + 1}/{len(CALIBRATION)}", flush=True)
    REF.linear = exact
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({f"{k[0]}:{k[1]}": v for k, v in amax.items()}, a.out)
    print("saved", a.out, len(amax), "groups", flush=True)


if __name__ == "__main__":
    main()
