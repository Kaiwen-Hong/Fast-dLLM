#!/usr/bin/env python3
"""GPU/TPU parity harness for eval_sasd (B4 rehearsal).

Runs the PRODUCTION TPU inference path: text decoder + section-diffusion sampler with PRECOMPUTED
image embeds (the ViT runs offline; the pod/GPU sampler consumes stored embeds via _EmbedShim — same
as eval_sasd.driver). Dumps the FULL denoised token sequence + parsed trajectory + decoded JSON per
sample, so a comparator can diff GPU vs TPU at the token level.

Memory: run ONE sample per process (`--samples val_s00`). Looping many samples in one process leaks
device buffers across samples and OOMs the 32 GB 5090 — a fresh process per sample reclaims all VRAM
(each sample then behaves like the validated single-sample smoke).

  PYTHONPATH=<fork>/src python parity_eval.py \
    --snapshot <release HF snapshot> --npz_dir <eval_inputs> --samples val_s00 \
    --dtype fp32 --out gpu_val_s00_fp32.json --tokens_dir tokens_gpu
"""
from __future__ import annotations
import argparse, gc, json, os, re, time
import numpy as np
import ml_dtypes

NULL_ID, MASK_ID, IMAGE_ID = 151666, 151665, 151655


class _EmbedShim:
    """Stand-in for the ViT when embeds are precomputed: sampler calls vit(pv, thw) once -> returns
    the stored embeds verbatim (no ViT on device). Mirrors eval_sasd.driver._EmbedShim."""
    def __init__(self, embeds):
        self._embeds = embeds
    def __call__(self, pixel_values, image_grid_thw):
        return self._embeds


def _parse_traj(t: str):
    m = re.findall(r"\[\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\]", t)
    return [[float(a), float(b)] for a, b in m[:5]] if m else None


def _valid_json(t: str) -> bool:
    try:
        return bool(json.loads(t[t.find("{"): t.rfind("}") + 1]))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--samples", required=True, help="comma list (use ONE per process for memory)")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens_dir", default=None)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    dtype = jnp.float32 if args.dtype == "fp32" else jnp.bfloat16
    if dtype == jnp.float32:
        jax.config.update("jax_default_matmul_precision", "highest")   # true fp32 parity baseline
    print(f"[harness] jax={jax.__version__} dev={jax.devices()} dtype={args.dtype} "
          f"matmul={jax.config.jax_default_matmul_precision}", flush=True)

    from maxtext.diffusion.eval_sasd.hf_to_jax import load_fast_ddrive_text
    from maxtext.diffusion.eval_sasd.models.qwen2_5_text import Qwen25TextConfig
    from maxtext.diffusion.eval_sasd.sampler_sasd import mm_section_diffusion_sample, decode_generation
    from transformers import AutoTokenizer

    t0 = time.time()
    text, _ = load_fast_ddrive_text(args.snapshot, Qwen25TextConfig.fast_ddrive(dtype), dtype=dtype, verbose=False)
    print(f"[harness] text loaded ({time.time()-t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.snapshot, trust_remote_code=True)

    if args.tokens_dir:
        os.makedirs(args.tokens_dir, exist_ok=True)
    results = []
    for s in args.samples.split(","):
        s = s.strip()
        try:
            d = np.load(os.path.join(args.npz_dir, f"{s}.npz"))
            files = set(d.files)
            x_t0 = d["x_t0"]; rbi = d["rbi"]; pos = d["position_ids"]; orig_len = int(d["orig_len"])
            raw = d["image_embeds"]
            if raw.dtype.kind == "V":                       # bf16 does NOT survive np.savez -> |V2 void
                raw = raw.view(ml_dtypes.bfloat16)
            embeds = jnp.asarray(raw, dtype)
            vit = _EmbedShim(embeds)
            pv = np.zeros((1, 1), np.float16)               # placeholder; shim ignores it
            thw = np.asarray(d["image_grid_thw"]) if "image_grid_thw" in files else np.array([[1, 1, 1]])
            ts = time.time()
            out_raw = mm_section_diffusion_sample(text, vit, x_t0, rbi, pos, pv, thw,
                                                  threshold=args.threshold, dtype=dtype)
            txt = decode_generation(out_raw, orig_len, tok)
            out = np.asarray(out_raw).reshape(-1).astype(np.int64)
            traj = _parse_traj(txt)
            r = {
                "sample": s, "dtype": args.dtype,
                "n_tokens": int(out.shape[0]),
                "n_mask_remaining": int((out == MASK_ID).sum()),
                "n_image_tokens": int((np.asarray(x_t0).reshape(-1) == IMAGE_ID).sum()),
                "valid_json": _valid_json(txt),
                "traj_parseable": traj is not None and len(traj) == 5,
                "traj": traj,
                "gen_text": txt,
                "sec_s": round(time.time() - ts, 1),
            }
            if args.tokens_dir:
                np.save(os.path.join(args.tokens_dir, f"{s}_{args.dtype}.npy"), out)
            results.append(r)
            print(f"[harness] {s} {args.dtype}: mask_rem={r['n_mask_remaining']} "
                  f"valid_json={r['valid_json']} traj={'ok' if r['traj_parseable'] else 'NO'} "
                  f"({r['sec_s']}s)", flush=True)
            del out_raw, out
        except Exception as e:
            import traceback
            print(f"[harness] {s} FAILED: {e}\n{traceback.format_exc()}", flush=True)
            results.append({"sample": s, "dtype": args.dtype, "error": str(e)})
        gc.collect()
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"[harness] wrote {args.out} ({len(results)} samples)", flush=True)


if __name__ == "__main__":
    main()
