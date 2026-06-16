"""Frozen-ViT embedding parity / precomputed-embeds validation for the internal TPU.

DESIGN: the canonical inference path feeds PRECOMPUTED fp32->bf16 image embeds (the TPU skips the
ViT) — symmetric with training, which also feeds precomputed embeds, so train==infer with zero
ViT-recompute drift. This check's PRIMARY job is to confirm the shipped precomputed embeds reproduce
a fresh fp32 ViT; the bf16-ViT numbers are a DIAGNOSTIC (the drift you'd incur IF you ran the bf16
ViT on-device, which the canonical path does not).

Reports max-abs / max-REL / cosine for:
  1. fp32(ViT here) vs bf16(ViT here)              -> bf16-ViT-matmul drift            [DIAGNOSTIC]
  2. fp32(ViT here) vs the npz precomputed embeds   -> do the shipped embeds reproduce fp32?  [GATE
     when present; cosine ~1.0 expected]
  3. bf16(ViT here) vs the npz precomputed embeds   -> bf16-ViT vs shipped               [DIAGNOSTIC]

Verdict: gate on (2) fp32_vs_reference when the npz carries embeds (the canonical path); else fall
back to gating on (1) fp32_vs_bf16 (on-device path — prefer fp32 ViT; the BASE ViT's bf16 drift is
~0.998 cosine, LARGER than the release ViT's 0.99966). Cosine is the meaningful similarity; max_rel
is a loose secondary guard (a 32-layer ViT amplifies per-element max_rel at small-magnitude rows).
Defaults: cosine >= 0.999, max_rel <= 5e-2.

Self-contained (fork-only), exportable: only scalars leave. The ViT loader reads the bf16 snapshot.

Usage:
  PYTHONPATH=fork/src python -m maxtext.diffusion.eval_sasd.embedding_parity \
      --npz <inputs_with_pixels.npz> --snapshot <B1 HF snapshot> \
      [--ref_rel 5e-2] [--cos 0.999] [--vlog validation_log.jsonl]
"""
from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from .hf_to_jax import load_fast_ddrive_vit
from .models.vision_qwen25vl import VisionConfig


def _embeds(snapshot_dir, pixel_values, image_grid_thw, dtype):
    if dtype == jnp.float32:
        jax.config.update("jax_default_matmul_precision", "highest")
    vit, _ = load_fast_ddrive_vit(snapshot_dir, VisionConfig(dtype=dtype), dtype=dtype, verbose=False)
    e = vit(jnp.asarray(pixel_values, dtype), np.asarray(image_grid_thw))
    return np.asarray(e, np.float32)   # upcast for comparison only


def _cmp(a, b):
    md = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b))) + 1e-12
    cos = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {"max_abs": md, "max_rel": md / scale, "cosine": cos}


def run_parity(npz_path, snapshot_dir, *, rel_thresh=5e-2, cos_thresh=0.999,
               vlog_path=None, run_id="sasd-eval", verbose=True) -> dict:
    d = np.load(npz_path)
    assert "pixel_values" in d.files and "image_grid_thw" in d.files, \
        "embedding parity needs pixel_values + image_grid_thw in the npz"
    pv, thw = d["pixel_values"], d["image_grid_thw"]

    e_f32 = _embeds(snapshot_dir, pv, thw, jnp.float32)
    e_bf16 = _embeds(snapshot_dir, pv, thw, jnp.bfloat16)
    res = {"n_rows": int(e_f32.shape[0]), "dim": int(e_f32.shape[1]),
           "fp32_vs_bf16": _cmp(e_f32, e_bf16)}

    if "image_embeds" in d.files:                      # offline reference (bf16-stored, fp32-computed)
        ref = d["image_embeds"]
        if ref.dtype.kind == "V":                      # bf16 does NOT survive np.savez -> |V2 void bytes
            ref = ref.view(ml_dtypes.bfloat16)
        ref = np.asarray(ref, np.float32)
        if ref.shape == e_f32.shape:
            res["fp32_vs_reference"] = _cmp(e_f32, ref)
            res["bf16_vs_reference"] = _cmp(e_bf16, ref)

    # Verdict reflects the design (precomputed embeds = canonical path):
    #  - reference present -> gate on fp32_vs_reference (do the SHIPPED embeds reproduce a fresh fp32
    #    ViT? confirms they're legit + match training). fp32_vs_bf16 / bf16_vs_reference stay as
    #    reported DIAGNOSTICS (the bf16-ViT drift the canonical path never incurs).
    #  - no reference (on-device fallback) -> gate on fp32_vs_bf16 (prefer fp32 ViT; bf16 drifts).
    g = res["fp32_vs_bf16"]
    if "fp32_vs_reference" in res:
        ok = res["fp32_vs_reference"]["cosine"] >= cos_thresh
        res["gated_on"] = "fp32_vs_reference"
    else:
        ok = g["cosine"] >= cos_thresh and g["max_rel"] <= rel_thresh
        res["gated_on"] = "fp32_vs_bf16"
    res["verdict"] = "PASS" if ok else "FAIL"
    res["thresholds"] = {"max_rel": rel_thresh, "cosine": cos_thresh}

    if verbose:
        print(f"[parity] {json.dumps(res, indent=2)}", flush=True)
        gm = res.get("fp32_vs_reference", g)
        print(f"EMBED_PARITY_{res['verdict']} (gated on {res['gated_on']}: cosine={gm['cosine']:.5f}; "
              f"bf16-ViT diag: fp32_vs_bf16 cosine={g['cosine']:.5f}, max_rel={g['max_rel']:.2e})", flush=True)
    if vlog_path is not None:
        rec = {"run_id": run_id, "stage": "S6_tpu_parity", "event": "embed_parity",
               "metrics": res, "verdict": res["verdict"], "refs": {"snapshot": snapshot_dir, "npz": npz_path}}
        with open(vlog_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return res


def main():
    ap = argparse.ArgumentParser(description="Frozen-ViT embedding parity (fp32 vs bf16 vs reference)")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--ref_rel", type=float, default=5e-2,
                    help="bf16-vs-fp32 max_rel loose guard (cosine is the primary gate)")
    ap.add_argument("--cos", type=float, default=0.999, help="cosine pass threshold")
    ap.add_argument("--vlog", default=None)
    ap.add_argument("--run_id", default="sasd-eval")
    args = ap.parse_args()
    run_parity(args.npz, args.snapshot, rel_thresh=args.ref_rel, cos_thresh=args.cos,
               vlog_path=args.vlog, run_id=args.run_id)


if __name__ == "__main__":
    main()
