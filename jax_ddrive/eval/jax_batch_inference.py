"""Fast-dDrive JAX multimodal eval — the JAX counterpart of fast_ddrive/eval/batch_inference.py.

Consumes the per-sample npz from prep_jax_eval.py, runs the JAX ViT + section-diffusion
sampler (ddrive_jax/eval/mm_sampler.py) for each sample, parses the trajectory, and writes
predictions.json in the SAME schema as the PyTorch eval — so the SAME official metric
(fast_ddrive/eval/evaluate_waymo_metrics.py) scores both stacks.

Run (jax env, free GPU):
  XLA_PYTHON_CLIENT_PREALLOCATE=false python jax_ddrive/eval/jax_batch_inference.py \
      --prep_dir .../prep_val --out_dir .../jax_val_sd [--fp32]"""
import argparse, json, os, re, sys, time
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")


def parse_trajectory(text):
    """Extract up to 5 [x,y] waypoints from the trajectory section (mirrors batch_inference)."""
    blob = text
    if '"trajectory"' in text:
        blob = text[text.index('"trajectory"'):]
    pairs = re.findall(r'\[\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\]', blob)
    if not pairs:
        return None
    return [[float(a), float(b)] for a, b in pairs[:5]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fp32", action="store_true", help="fp32 (tighter parity, more memory); default bf16.")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()

    if args.fp32:
        jax.config.update("jax_default_matmul_precision", "highest")
    dt = jnp.float32 if args.fp32 else jnp.bfloat16

    from transformers import AutoTokenizer
    from ddrive_jax.models.qwen2_5_text import Qwen25TextConfig
    from ddrive_jax.models.vision_qwen25vl import VisionConfig
    from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_text, load_fast_ddrive_vit
    from ddrive_jax.eval.mm_sampler import mm_section_diffusion_sample, decode_generation

    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    print(f"loading JAX text + ViT ({dt.__name__}) ...", flush=True)
    text, _ = load_fast_ddrive_text(SNAP, Qwen25TextConfig.fast_ddrive(dt), dtype=dt, verbose=False)
    vit, _ = load_fast_ddrive_vit(SNAP, VisionConfig(dtype=dt), dtype=dt, verbose=False)

    manifest = json.load(open(os.path.join(args.prep_dir, "manifest.json")))
    if args.max_samples > 0:
        manifest = manifest[: args.max_samples]
    os.makedirs(args.out_dir, exist_ok=True)

    preds = []
    t0 = time.time()
    for i, m in enumerate(manifest):
        d = np.load(os.path.join(args.prep_dir, m["npz"]))
        sid = str(d["sample_id"]); orig_len = int(d["orig_len"])
        try:
            out = mm_section_diffusion_sample(
                text, vit, d["x_t0"], d["rbi"], d["position_ids"],
                d["pixel_values"].astype(np.float32), d["image_grid_thw"],
                threshold=args.threshold, dtype=dt)
            raw = decode_generation(out, orig_len, tok)
            traj = parse_trajectory(raw)
            err = None
        except Exception as e:
            import traceback; raw = None; traj = None; err = traceback.format_exc()[-300:]
        gt = d["future_waypoints"].tolist() if d["future_waypoints"].size else []
        preds.append({"sample_id": sid, "pred_trajectory": traj, "model_output_raw": raw,
                      "future waypoints": gt, "error": err})
        if (i + 1) % 10 == 0 or i == len(manifest) - 1:
            ok = sum(1 for p in preds if p["pred_trajectory"])
            print(f"  [{i+1}/{len(manifest)}] parsed={ok} ({(time.time()-t0)/(i+1):.2f}s/sample)", flush=True)

    n_ok = sum(1 for p in preds if p["pred_trajectory"])
    out = {"metadata": {"stack": "jax", "dtype": dt.__name__, "mode": "section_diffusion",
                        "threshold": args.threshold, "elapsed_s": round(time.time() - t0, 1),
                        "n_samples": len(preds)},
           "aggregate_metrics": {"total_samples": len(preds), "valid_trajectory": n_ok,
                                 "valid_trajectory_ratio": n_ok / max(len(preds), 1)},
           "predictions": preds}
    pj = os.path.join(args.out_dir, "predictions.json")
    json.dump(out, open(pj, "w"))
    print(f"\n[jax-eval] {n_ok}/{len(preds)} valid trajectories → {pj}")
    print("JAX_BATCH_INFERENCE_DONE")


if __name__ == "__main__":
    main()
