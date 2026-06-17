#!/usr/bin/env python3
"""Build the GPU<->TPU parity report from parity_eval.py outputs.

- GPU sensitivity: fp32 vs bf16 token agreement (is the test even sensitive to numerics?).
- GPU vs TPU: per precision, token exact-match + agreement + first divergence + trajectory max|Δ|.
Emits a markdown report + a json. Token arrays read from each side's tokens_dir as {sample}_{dtype}.npy.

  python parity_report.py --gpu_dir <temp> --tpu_tag r1 --out_md report.md --out_json report.json
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np


def load_json(p):
    try:
        return {r["sample"]: r for r in json.load(open(p))}
    except Exception:
        return {}


def tok(d, s, dt):
    p = os.path.join(d, f"{s}_{dt}.npy")
    return np.load(p).reshape(-1) if os.path.exists(p) else None


def cmp_tokens(a, b):
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    eq = a[:n] == b[:n]
    di = np.where(~eq)[0]
    return {"same_len": bool(len(a) == len(b)), "n": int(n),
            "exact": bool(eq.all() and len(a) == len(b)),
            "agree": round(float(eq.mean()), 6),
            "first_div": int(di[0]) if di.size else None,
            "n_div": int((~eq).sum())}


def traj_delta(a, b):
    if not a or not b:
        return None
    a, b = np.array(a, float), np.array(b, float)
    return float(np.abs(a - b).max()) if a.shape == b.shape else None


def pair(A, B, ad, bd, dt, samples):
    rows, exact, trajok = [], 0, 0
    for s in samples:
        ta, tb = tok(ad, s, dt), tok(bd, s, dt)
        c = cmp_tokens(ta, tb)
        td = traj_delta(A.get(s, {}).get("traj"), B.get(s, {}).get("traj"))
        r = {"sample": s, **(c or {"note": "missing tokens"}),
             "traj_max_delta": td, "traj_le_0.1m": (td is not None and td <= 0.1),
             "text_equal": A.get(s, {}).get("gen_text") == B.get(s, {}).get("gen_text")}
        if c and c["exact"]:
            exact += 1
        if r["traj_le_0.1m"]:
            trajok += 1
        rows.append(r)
    return {"n": len(samples), "token_exact": exact, "traj_le_0.1m": trajok,
            "text_equal": sum(1 for r in rows if r.get("text_equal")),
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_dir", required=True)
    ap.add_argument("--tpu_tag", default=None, help="e.g. r1 -> tpu_r1_{fp32,bf16}.json + tokens_tpu_r1")
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    W = args.gpu_dir

    G = {dt: load_json(os.path.join(W, f"gpu_{dt}.json")) for dt in ("fp32", "bf16")}
    gtok = os.path.join(W, "tokens_gpu")
    rep = {"gpu_ok": {dt: sum("error" not in r for r in G[dt].values()) for dt in G}}

    # GPU internal sensitivity: fp32 vs bf16 (same dir, different dtype suffix)
    g_samps = sorted(set(G["fp32"]) & set(G["bf16"]))
    sens = []
    for s in g_samps:
        c = cmp_tokens(tok(gtok, s, "fp32"), tok(gtok, s, "bf16"))
        if c:
            sens.append(c["agree"])
    rep["gpu_fp32_vs_bf16"] = {"n": len(sens),
                               "mean_token_agree": round(float(np.mean(sens)), 4) if sens else None,
                               "min_token_agree": round(float(np.min(sens)), 4) if sens else None,
                               "interpretation": ("sensitive (numerics affect tokens) — GPU<->TPU test is meaningful"
                                                  if sens and np.mean(sens) < 0.999 else
                                                  "near-identical — model is confident; GPU<->TPU agreement may be trivial")}

    # GPU vs TPU per precision
    if args.tpu_tag:
        for dt in ("fp32", "bf16"):
            T = load_json(os.path.join(W, f"tpu_{args.tpu_tag}_{dt}.json"))
            ttok = os.path.join(W, f"tokens_tpu_{args.tpu_tag}")
            samples = sorted(set(G[dt]) & set(T))
            rep[f"gpu_vs_tpu_{dt}"] = pair(G[dt], T, gtok, ttok, dt, samples)
            rep[f"gpu_vs_tpu_{dt}"]["verdict"] = (
                "PARITY" if rep[f"gpu_vs_tpu_{dt}"]["token_exact"] == len(samples) and samples
                else "DIVERGENCE")

    json.dump(rep, open(args.out_json, "w"), indent=2)

    # markdown
    L = ["# GPU<->TPU eval_sasd parity report", ""]
    L.append(f"GPU samples ok: fp32={rep['gpu_ok']['fp32']}, bf16={rep['gpu_ok']['bf16']}")
    s = rep["gpu_fp32_vs_bf16"]
    L += ["", "## GPU internal sensitivity (fp32 vs bf16)",
          f"- n={s['n']}, mean token-agree={s['mean_token_agree']}, min={s['min_token_agree']}",
          f"- {s['interpretation']}"]
    for dt in ("fp32", "bf16"):
        k = f"gpu_vs_tpu_{dt}"
        if k in rep:
            r = rep[k]
            L += ["", f"## GPU vs TPU — {dt}  [{r['verdict']}]",
                  f"- n={r['n']} | token-exact={r['token_exact']}/{r['n']} | "
                  f"traj≤0.1m={r['traj_le_0.1m']}/{r['n']} | text-equal={r['text_equal']}/{r['n']}",
                  "", "| sample | exact | agree | first_div | traj_Δ | text_eq |",
                  "|---|---|---|---|---|---|"]
            for x in r["rows"]:
                L.append(f"| {x['sample']} | {x.get('exact')} | {x.get('agree')} | "
                         f"{x.get('first_div')} | {x.get('traj_max_delta')} | {x.get('text_equal')} |")
    open(args.out_md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:18]))
    print(f"\n[report] wrote {args.out_md} + {args.out_json}")


if __name__ == "__main__":
    main()
