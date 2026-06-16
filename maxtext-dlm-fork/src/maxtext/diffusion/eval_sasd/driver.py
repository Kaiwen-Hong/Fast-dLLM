"""SASD inference driver — self-contained, fork-only, internal-TPU ready (blocker B2).

Closes B2: runs the validated multimodal section-diffusion sampler against a B1-exported
HF snapshot, entirely inside the MaxText fork (PYTHONPATH=fork/src; no ddrive_jax). The
trained model is produced by MaxText -> Orbax param ckpt -> `maxtext_to_hf_export.py` (B1,
bf16 HF snapshot) -> the bf16-hardened NNX loaders here -> this sampler.

Input is a single npz produced offline by `scripts/prep_jax_eval_inputs.py` (so the internal
TPU needs no torch / HF processor / ViT — only an AutoTokenizer to decode the output). The npz
carries the inference scaffold and, preferably, the precomputed frozen-ViT image embeds:
  required : x_t0[int64 L], rbi[int32 L], position_ids[int32 3xL], orig_len[int]
  image    : image_embeds[bf16 N,D]  (preferred — TPU skips the ViT)
             OR pixel_values[f16] + image_grid_thw[int64]  (ViT runs on-device)
  optional : target_ids[int64]  (the GT answer tokens, for T2 exact-match scoring)

Only SCALAR results leave the box (validation_log.jsonl): valid_json, traj_parseable,
traj_exact (vs target), token_agreement, max trajectory |Δ| vs the reference. No weights,
images, or raw sample text are emitted.

Usage (fork env; fp32 first, then bf16 — see the user's both-precisions policy):
  PYTHONPATH=/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/src \
  python -m maxtext.diffusion.eval_sasd.driver \
      --npz <inputs.npz> --snapshot <B1 HF snapshot> --dtype fp32 \
      [--tokenizer <dir>] [--vlog validation_log.jsonl] [--run_id overfit400-base-r1]
"""
from __future__ import annotations

import argparse
import json
import re

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from .hf_to_jax import load_fast_ddrive_text, load_fast_ddrive_vit
from .models.qwen2_5_text import Qwen25TextConfig
from .models.vision_qwen25vl import VisionConfig
from .sampler_sasd import mm_section_diffusion_sample, decode_generation

NULL_ID, MASK_ID, IMAGE_ID = 151666, 151665, 151655


class _EmbedShim:
    """Stand-in for the ViT when image embeds are already precomputed: the sampler calls
    ``vit(pixel_values, image_grid_thw)`` once (sampler_sasd.py); this returns the stored
    embeds verbatim so the on-device ViT is never loaded/run."""

    def __init__(self, embeds):
        self._embeds = embeds

    def __call__(self, pixel_values, image_grid_thw):  # signature matches VisionTransformer.__call__
        return self._embeds


def _parse_trajectory(text: str):
    """Extract up to 5 [x,y] waypoints from the decoded JSON text (mirrors verify_sd_mm.py)."""
    m = re.findall(r"\[\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\]", text)
    return np.array([[float(a), float(b)] for a, b in m[:5]]) if m else None


def _valid_json(text: str) -> bool:
    try:
        return bool(json.loads(text[text.find("{"): text.rfind("}") + 1]))
    except Exception:
        return False


def _sections(text: str):
    """Parse the 4-section SASD JSON object from decoded text; return the dict or None."""
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def run_eval(npz_path: str, snapshot_dir: str, *, dtype=jnp.float32, threshold: float = 0.9,
             tokenizer_dir: str | None = None, vlog_path: str | None = None,
             run_id: str = "sasd-eval", sample_tag: str | None = None,
             show_text: bool = False, verbose: bool = True) -> dict:
    """Generate one sample and return scalar metrics; append a vlog event if vlog_path given."""
    from transformers import AutoTokenizer

    if dtype == jnp.float32:
        jax.config.update("jax_default_matmul_precision", "highest")  # true fp32 (parity baseline)

    d = np.load(npz_path)
    files = set(d.files)
    x_t0 = d["x_t0"]; rbi = d["rbi"]; pos = d["position_ids"]; orig_len = int(d["orig_len"])

    text, _ = load_fast_ddrive_text(
        snapshot_dir, Qwen25TextConfig.fast_ddrive(dtype), dtype=dtype, verbose=False)

    if "image_embeds" in files:                       # preferred: TPU skips the ViT
        raw = d["image_embeds"]
        if raw.dtype.kind == "V":                     # bf16 does NOT survive np.savez -> |V2 void bytes
            raw = raw.view(ml_dtypes.bfloat16)
        embeds = jnp.asarray(raw, dtype)
        vit = _EmbedShim(embeds)
        pv = np.zeros((1, 1), np.float16)             # placeholder; the shim ignores it
        thw = np.asarray(d["image_grid_thw"]) if "image_grid_thw" in files else np.array([[1, 1, 1]])
        img_src = "precomputed_embeds"
    else:                                             # fallback: run the ViT on-device
        vit, _ = load_fast_ddrive_vit(snapshot_dir, VisionConfig(dtype=dtype), dtype=dtype, verbose=False)
        pv = d["pixel_values"]; thw = d["image_grid_thw"]
        img_src = "on_device_vit"

    if verbose:
        print(f"[eval] generating (L={x_t0.shape[0]}, dtype={dtype.__name__}, img={img_src}) ...", flush=True)
    out = mm_section_diffusion_sample(text, vit, x_t0, rbi, pos, pv, thw,
                                      threshold=threshold, dtype=dtype)

    tok = AutoTokenizer.from_pretrained(tokenizer_dir or snapshot_dir, trust_remote_code=True)
    gen_txt = decode_generation(out, orig_len, tok)
    tj = _parse_trajectory(gen_txt)

    metrics = {
        "dtype": dtype.__name__,
        "img_source": img_src,
        "valid_json": _valid_json(gen_txt),
        "traj_parseable": tj is not None and tj.shape == (5, 2),
    }
    # scalar diagnostics — catch resolution/scaffold/decode corruption without exporting any text
    out_flat = np.asarray(out).reshape(-1)
    metrics["L"] = int(np.asarray(x_t0).shape[0])
    metrics["n_image_tokens"] = int((np.asarray(x_t0) == IMAGE_ID).sum())  # expect 168 at train res
    metrics["n_mask_remaining"] = int((out_flat == MASK_ID).sum())          # expect 0 (fully denoised)

    # Optional GT-target comparison (T2 memorisation). NOTE: a positional token-by-token "agreement"
    # between the denoised scaffold (interleaved NULL/MASK/structural tokens) and the FLAT GT answer
    # tokenisation is meaningless — the indices don't line up — so we score structure-aware decoded
    # quantities instead: numeric trajectory match + per-section (CO / FMB) exact equality.
    if "target_ids" in files:
        tgt = np.asarray(d["target_ids"]).reshape(-1)
        tgt_txt = tok.decode([int(t) for t in tgt if t not in (NULL_ID, MASK_ID)], skip_special_tokens=True)
        tjt = _parse_trajectory(tgt_txt)
        metrics["traj_exact"] = bool(tj is not None and tjt is not None
                                     and tj.shape == tjt.shape and np.array_equal(tj, tjt))
        metrics["traj_max_abs_delta"] = (float(np.abs(tj - tjt).max())
                                         if (tj is not None and tjt is not None and tj.shape == tjt.shape)
                                         else None)
        gj, tjs = _sections(gen_txt), _sections(tgt_txt)
        metrics["co_match"] = bool(gj is not None and tjs is not None
                                   and gj.get("critical_objects") == tjs.get("critical_objects"))
        metrics["fmb_match"] = bool(gj is not None and tjs is not None
                                    and gj.get("future_meta_behavior") == tjs.get("future_meta_behavior"))
    elif "ref_output" in files:                       # oracle-style reference (local parity test)
        ref = np.asarray(d["ref_output"]).reshape(-1)
        gen = np.asarray(out).reshape(-1)[orig_len:]
        n = min(len(gen), len(ref) - orig_len)
        metrics["token_agreement"] = float((gen[:n] == ref[orig_len:orig_len + n]).mean()) if n else 0.0

    if verbose:
        print(f"[eval] {metrics}", flush=True)
        if show_text:                                 # raw model text must NOT leave the box by default
            print(f"[eval] generated:\n{gen_txt[:500]}", flush=True)

    if vlog_path is not None:
        _emit_vlog(vlog_path, run_id, "S4_infer", "infer_sample",
                   {**metrics, "sample": sample_tag}, threshold=threshold,
                   refs={"snapshot": snapshot_dir, "npz": npz_path})
    return metrics


def _emit_vlog(path, run_id, stage, event, metrics, *, threshold=None, verdict="INFO", refs=None):
    """Append one exportable validation-log event (scalars/bools/hashes only) — blocker doc §7."""
    rec = {"run_id": run_id, "stage": stage, "event": event,
           "metrics": metrics, "threshold": threshold, "verdict": verdict, "refs": refs or {}}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser(description="SASD inference driver (fork-only, internal-TPU ready)")
    ap.add_argument("--npz", required=True, help="inference inputs npz (prep_jax_eval_inputs.py)")
    ap.add_argument("--snapshot", required=True, help="B1-exported HF snapshot dir (bf16)")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--vlog", default=None)
    ap.add_argument("--run_id", default="sasd-eval")
    ap.add_argument("--sample", default=None, help="sample tag/hash for the vlog")
    ap.add_argument("--show_text", action="store_true",
                    help="also print 500 chars of generated text (LOCAL DEBUG ONLY — off for the "
                         "internal scalar-only export)")
    args = ap.parse_args()
    dt = jnp.float32 if args.dtype == "fp32" else jnp.bfloat16
    m = run_eval(args.npz, args.snapshot, dtype=dt, threshold=args.threshold,
                 tokenizer_dir=args.tokenizer, vlog_path=args.vlog, run_id=args.run_id,
                 sample_tag=args.sample, show_text=args.show_text)
    ok = m["valid_json"] and m["traj_parseable"]
    print(f"SASD_EVAL_{'PASS' if ok else 'FAIL'} (valid JSON + 5-waypoint trajectory)")


if __name__ == "__main__":
    main()
