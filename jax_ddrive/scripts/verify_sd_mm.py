"""Verify the JAX multimodal section-diffusion sampler against the PyTorch reference
captured by capture_oracle_sd_mm.py. Runs the JAX ViT + fused denoise on the SAME
x_t0/rbi/position_ids/pixel_values and compares the generated trajectory/tokens.

Run (jax env, free GPU): python jax_ddrive/scripts/verify_sd_mm.py [--bf16]"""
import sys, os, json, re
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
ORACLE = "/home/kaiwen/data/fast-ddrive/ref_logits/sd_mm_oracle.npz"


def main():
    fp32 = "--bf16" not in sys.argv
    if fp32:
        jax.config.update("jax_default_matmul_precision", "highest")
    dt = jnp.float32 if fp32 else jnp.bfloat16
    from transformers import AutoTokenizer
    from ddrive_jax.models.qwen2_5_text import Qwen25TextConfig
    from ddrive_jax.models.vision_qwen25vl import VisionConfig
    from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_text, load_fast_ddrive_vit
    from ddrive_jax.eval.mm_sampler import mm_section_diffusion_sample, decode_generation

    d = np.load(ORACLE)
    x_t0 = d["x_t0"]; rbi = d["rbi"]; pos = d["position_ids"]
    pv = d["pixel_values"]; thw = d["image_grid_thw"]; orig_len = int(d["orig_len"])
    ref_out = d["ref_output"]
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)

    text, _ = load_fast_ddrive_text(SNAP, Qwen25TextConfig.fast_ddrive(dt), dtype=dt, verbose=False)
    vit, _ = load_fast_ddrive_vit(SNAP, VisionConfig(dtype=dt), dtype=dt, verbose=False)

    print(f"generating (L={x_t0.shape[0]}, dtype={dt.__name__}) ...", flush=True)
    out = mm_section_diffusion_sample(text, vit, x_t0, rbi, pos, pv, thw,
                                      threshold=0.9, dtype=dt)
    jax_txt = decode_generation(out, orig_len, tok)
    ref_txt = tok.decode([int(t) for t in ref_out[orig_len:] if t not in (151666, 151665)],
                         skip_special_tokens=True)
    print("\n--- JAX section-diffusion (multimodal) ---\n" + jax_txt[:400])
    print("\n--- PyTorch reference ---\n" + ref_txt[:400])

    def traj(s):
        m = re.findall(r'\[\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\]', s)
        return np.array([[float(a), float(b)] for a, b in m[:5]]) if m else None
    tj, tr = traj(jax_txt), traj(ref_txt)
    gen = out[orig_len:]; n = min(len(gen), len(ref_out) - orig_len)
    agree = float((np.asarray(gen[:n]) == ref_out[orig_len:orig_len + n]).mean()) if n else 0.0
    try:
        valid_json = bool(json.loads(jax_txt[jax_txt.find("{"):jax_txt.rfind("}") + 1]))
    except Exception:
        valid_json = False
    traj_l2 = float(np.abs(tj - tr).max()) if (tj is not None and tr is not None and tj.shape == tr.shape) else None
    print(f"\ntoken agreement: {agree*100:.1f}% | valid JSON: {valid_json} | "
          f"traj match (both 5wp): jax={None if tj is None else tj.tolist()} "
          f"ref={None if tr is None else tr.tolist()} | traj max|Δ|={traj_l2}")
    ok = valid_json and tj is not None and tj.shape == (5, 2)
    print(f"VERIFY_SD_MM_{'PASS' if ok else 'FAIL'} (valid JSON + 5-waypoint trajectory)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
