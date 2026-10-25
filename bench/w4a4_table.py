"""The MXFP4 GEMM's tuning table (kernels/w4a4.py pick): the engine's FP4 projection shapes (the MLP's gate|up and
down, and the full-FP4 path's projections when asked) at the engine's row counts, timed in every tile config; the
fastest per (shape, rows, epilogue) goes into the table.  BM and BN pick which program computes an element; the K
order (BK) is fixed, so every config gives the same bits (--verify checks it).

    python bench/w4a4_table.py                                   # the 9B's MLP shapes into
                                                                 # kernels/w4a4_table_<gpu>.json
    python bench/w4a4_table.py --shapes 24576x4096,4096x12288 --epi 0,1 --rows 1,32,128,2048,4096
    python bench/w4a4_table.py --recheck                         # table choices against the rule, at each entry

STARTLUX_PRISM_W4A4_TABLE names another table (the engine reads it too); "off" leaves the rule in charge.
"""
import argparse
import json
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from startlux_prism.kernels import cuda_ext  # noqa: E402
from startlux_prism.kernels import w4a4 as W  # noqa: E402

ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
# (BM, BN, GROUP_M, warps, stages): the 9B's MLP shapes on the RTX 5060 Ti were measured over these
CFGS = (
    (16, 64, 8, 4, 3), (32, 64, 8, 4, 3), (32, 128, 8, 4, 3), (64, 32, 8, 4, 3), (64, 64, 8, 4, 3),
    (64, 128, 8, 8, 3), (64, 128, 8, 8, 4), (64, 256, 8, 8, 3), (128, 64, 8, 4, 3), (128, 128, 8, 8, 3),
    (128, 128, 8, 8, 4), (128, 256, 8, 8, 3), (256, 128, 8, 8, 3), (256, 256, 8, 8, 3),
)


def make(n, k, m, epi, dev):
    g = torch.Generator(device=dev).manual_seed(n * 131 + k * 17 + m)
    q = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev, generator=g)
    s = torch.full((n, k // 32), 127, dtype=torch.uint8, device=dev)
    aq = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device=dev, generator=g)
    as_ = torch.full((m, k // 32), 127, dtype=torch.uint8, device=dev)
    out = torch.zeros((m, n), dtype=torch.float32 if epi else torch.bfloat16, device=dev)
    return W.MX4(q, s, n, k), aq, as_, out


def run(cfg, a_q, a_s, w, out, epi, m):
    bm, bn, group, warps, stages = cfg
    bk = 128 if w.k % 128 == 0 else 32
    grid = (triton.cdiv(m, bm) * triton.cdiv(w.n, bn),)
    W._gemm_kernel[grid](a_q, a_s, w.q, w.s, out, out, m, N=w.n, OUTS=out.stride(0), K=w.k, BM=bm, BN=bn, BK=bk,
                         GROUP_M=group, EPI=epi, num_warps=warps, num_stages=stages)


def timed(cfg, a_q, a_s, w, out, epi, m):
    fn = lambda: run(cfg, a_q, a_s, w, out, epi, m)  # noqa: E731
    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=10, rep=50)          # ms


def shapes_arg(text):
    out = []
    for part in text.split(","):
        n, k = (int(v) for v in part.lower().split("x"))
        out.append((n, k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="24576x4096,4096x12288", help="NxK projections (the 9B's gate|up and down)")
    ap.add_argument("--epi", default="0,1", help="epilogues to time: 0 (bf16 out), 1 (fp32 residual add)")
    ap.add_argument("--rows", default=",".join(str(r) for r in ROWS))
    ap.add_argument("--write", default="", help="the engine's table (default: kernels/w4a4_table_<gpu tag>.json)")
    ap.add_argument("--recheck", action="store_true", help="time the file's choices against the rule and keep the faster")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA is needed")
    path = a.write or cuda_ext.table_path("w4a4_table")
    rows = [int(v) for v in a.rows.split(",")]

    if a.recheck:
        doc = json.load(open(path))
        table = doc["table"]
        changed = 0
        for key, by_m in sorted(table.items()):
            n, k, epi = (int(v) for v in key.split(","))
            for m in sorted(int(v) for v in by_m):
                w, aq, as_, out = make(n, k, m, epi, "cuda")
                table_cfg = tuple(int(v) for v in by_m[str(m)])
                rule = W.pick_rule(m, n, k)
                rule_cfg = (rule[0], rule[1], rule[3], rule[4], rule[5])
                t_table, t_rule = timed(table_cfg, aq, as_, w, out, epi, m), timed(rule_cfg, aq, as_, w, out, epi, m)
                print(f"{key:18s} M {m:5d}  table {t_table:7.2f} ms  rule {t_rule:7.2f} ms  "
                      f"{'table' if t_table <= t_rule else 'rule'}", flush=True)
                if t_rule < t_table * 0.98:
                    table[key][str(m)] = list(rule_cfg)
                    changed += 1
                del w, aq, as_, out
                torch.cuda.empty_cache()
        json.dump(doc, open(path, "w"), indent=1)
        print(f"{path}: {changed} entries changed")
        return

    table = {}
    for n, k in shapes_arg(a.shapes):
        for epi in (int(v) for v in a.epi.split(",")):
            key = f"{n},{k},{epi}"
            table[key] = {}
            for m in rows:
                w, aq, as_, out = make(n, k, m, epi, "cuda")
                times = [(timed(c, aq, as_, w, out, epi, m), c) for c in CFGS]
                t_best, best = min(times, key=lambda t: t[0])
                rule = W.pick_rule(m, n, k)
                t_rule = timed((rule[0], rule[1], rule[3], rule[4], rule[5]), aq, as_, w, out, epi, m)
                if not a.no_verify:
                    ref, res = torch.zeros_like(out), torch.zeros_like(out)
                    run(CFGS[0], aq, as_, w, ref, epi, m)
                    run(best, aq, as_, w, res, epi, m)
                    if not torch.equal(ref, res):
                        raise SystemExit(f"{key} M {m}: {best} differs from {CFGS[0]}")
                table[key][str(m)] = list(best)
                print(f"{key:18s} M {m:5d}  {best}  {t_best:7.2f} ms  rule {t_rule:7.2f} ms  "
                      f"{t_rule / t_best:.2f}x", flush=True)
                del w, aq, as_, out
                torch.cuda.empty_cache()
    doc = {"gpu": torch.cuda.get_device_name(), "source": "bench/w4a4_table.py", "table": table}
    json.dump(doc, open(path, "w"), indent=1)
    print(f"wrote {path}: {len(table)} shapes")


if __name__ == "__main__":
    main()
