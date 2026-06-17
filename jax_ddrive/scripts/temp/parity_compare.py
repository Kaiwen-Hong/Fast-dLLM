#!/usr/bin/env python3
"""Compare two eval_sasd parity runs (e.g. GPU vs TPU) produced by parity_eval.py.

Diffs, per sample and per dtype:
  - token-level: exact-match bool, agreement fraction, index of FIRST divergence (the cascade origin)
  - trajectory : max |Δ| between the two parsed trajectories (target <= 0.1 m)
  - JSON       : decoded-text equality + valid_json on both sides
Reads the per-sample denoised-token .npy from each side's tokens_dir, plus the result jsons.

  python parity_compare.py --a_json gpu_fp32.json --a_tokens tokens_gpu \
                           --b_json tpu_fp32.json --b_tokens tokens_tpu \
                           --dtype fp32 --out compare_fp32.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


def _load_json(p):
    return {r["sample"]: r for r in json.load(open(p))}


def _traj_max_delta(a, b):
    if not a or not b:
        return None
    a = np.array(a, float); b = np.array(b, float)
    if a.shape != b.shape:
        return None
    return float(np.abs(a - b).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_json", required=True); ap.add_argument("--a_tokens", required=True)
    ap.add_argument("--b_json", required=True); ap.add_argument("--b_tokens", required=True)
    ap.add_argument("--a_name", default="A"); ap.add_argument("--b_name", default="B")
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A = _load_json(args.a_json); B = _load_json(args.b_json)
    samples = sorted(set(A) & set(B))
    rows = []
    n_exact = 0; n_traj_ok = 0; worst_first_div = None
    for s in samples:
        a, b = A[s], B[s]
        row = {"sample": s, "dtype": args.dtype}
        # token-level
        ta = os.path.join(args.a_tokens, f"{s}_{args.dtype}.npy")
        tb = os.path.join(args.b_tokens, f"{s}_{args.dtype}.npy")
        if os.path.exists(ta) and os.path.exists(tb):
            xa = np.load(ta).reshape(-1); xb = np.load(tb).reshape(-1)
            n = min(len(xa), len(xb))
            eq = xa[:n] == xb[:n]
            row["n_tokens"] = int(n)
            row["same_len"] = bool(len(xa) == len(xb))
            row["token_exact"] = bool(eq.all() and len(xa) == len(xb))
            row["token_agree"] = round(float(eq.mean()), 6)
            diff_idx = np.where(~eq)[0]
            row["first_divergence"] = int(diff_idx[0]) if diff_idx.size else None
            row["n_divergent_tokens"] = int((~eq).sum())
            if row["token_exact"]:
                n_exact += 1
            if row["first_divergence"] is not None:
                worst_first_div = (row["first_divergence"] if worst_first_div is None
                                   else min(worst_first_div, row["first_divergence"]))
        else:
            row["token_note"] = f"missing tokens ({'A' if not os.path.exists(ta) else ''}{'B' if not os.path.exists(tb) else ''})"
        # trajectory
        row["traj_max_delta"] = _traj_max_delta(a.get("traj"), b.get("traj"))
        row["traj_le_0.1m"] = (row["traj_max_delta"] is not None and row["traj_max_delta"] <= 0.1)
        if row["traj_le_0.1m"]:
            n_traj_ok += 1
        # json / text
        row["valid_json"] = {args.a_name: a.get("valid_json"), args.b_name: b.get("valid_json")}
        row["text_equal"] = (a.get("gen_text") == b.get("gen_text"))
        row["mask_remaining"] = {args.a_name: a.get("n_mask_remaining"), args.b_name: b.get("n_mask_remaining")}
        rows.append(row)

    summary = {
        "dtype": args.dtype, "a": args.a_name, "b": args.b_name,
        "n_samples": len(samples),
        "token_exact": n_exact, "traj_le_0.1m": n_traj_ok,
        "text_equal": sum(1 for r in rows if r.get("text_equal")),
        "earliest_first_divergence": worst_first_div,
        "verdict": "PARITY" if n_exact == len(samples) and len(samples) > 0 else "DIVERGENCE",
    }
    out = {"summary": summary, "rows": rows}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    # compact per-sample line
    for r in rows:
        print(f"  {r['sample']}: exact={r.get('token_exact')} agree={r.get('token_agree')} "
              f"first_div={r.get('first_divergence')} traj_Δ={r.get('traj_max_delta')} "
              f"text_eq={r.get('text_equal')}")


if __name__ == "__main__":
    main()
