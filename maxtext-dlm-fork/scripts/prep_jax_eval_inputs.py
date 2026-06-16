"""OFFLINE: build the SASD inference-input npz for the internal-TPU eval (blocker B2/B3).

Runs LOCALLY (or on the free-TPU host), where torch + the HF processor + ddrive_jax exist.
Produces a single npz per sample that the fork-only driver (`maxtext.diffusion.eval_sasd.driver`)
consumes on the internal TPU — so the internal side needs NO torch / processor / ViT, only an
AutoTokenizer to decode the output.

This is the input half of capture_oracle_sd_mm.py with the PyTorch model removed: the scaffold
and rope index come from the JAX-native, parity-validated `build_scaffold` /
`get_rope_index_numpy` (capture_oracle validates these match the PyTorch internals bit-for-bit).

npz contents (matches driver.run_eval):
  x_t0[int64 L], rbi[int64 L], position_ids[int64 3xL], orig_len[int64],
  pixel_values[f32], image_grid_thw[int64],
  target_ids[int64]            (tokenized GT answer, for T2 exact-match)
  image_embeds[bf16 N,D]       (only with --with_embeds: frozen ViT run offline so the TPU
                                skips the ViT; also the reference for the embedding-parity check)

NOTE: this tool intentionally imports ddrive_jax (offline only). The internal-TPU INFERENCE
path (eval_sasd) imports NO ddrive_jax — that separation is the whole point of B2.

Usage (jax/ddrive env, local):
  python scripts/prep_jax_eval_inputs.py --sample sample.json --img_dir <jpgs> \
      --snapshot <base or release HF snapshot> --out inputs.npz [--with_embeds] [--idx 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# ddrive_jax is available OFFLINE; add its root if not already importable.
_DDRIVE_ROOT = "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive"
if _DDRIVE_ROOT not in sys.path:
    sys.path.insert(0, _DDRIVE_ROOT)


def main():
    ap = argparse.ArgumentParser(description="Offline SASD inference-input npz builder (B2/B3)")
    ap.add_argument("--sample", required=True, help="eval sample JSON (list of {conversations,image})")
    ap.add_argument("--img_dir", required=True, help="dir holding the per-frame camera JPEGs")
    ap.add_argument("--snapshot", required=True, help="HF snapshot for tokenizer/processor (base or release)")
    ap.add_argument("--out", required=True, help="output npz path")
    ap.add_argument("--idx", type=int, default=0, help="which sample in the JSON list")
    ap.add_argument("--with_embeds", action="store_true",
                    help="also run the frozen ViT offline and store bf16 image_embeds (TPU skips ViT)")
    # MUST match the resolution the model was TRAINED at, or the image-token count / mRoPE
    # positions won't line up. Training (prep_train_jax.py) used min=784, max=784*64 (~64 merged
    # tokens/img → grid [1,16,14] = 56/img). For paper-eval inputs instead, pass 200704/200704.
    ap.add_argument("--min_pixels", type=int, default=784, help="match training prep (default 784)")
    ap.add_argument("--max_pixels", type=int, default=784 * 64, help="match training prep (default 50176)")
    args = ap.parse_args()

    import torch  # offline only
    from transformers import AutoProcessor, AutoTokenizer
    from PIL import Image
    from ddrive_jax.eval.scaffold import build_scaffold, messages_from_prompt
    from ddrive_jax.eval.rope_index import get_rope_index_numpy

    tok = AutoTokenizer.from_pretrained(args.snapshot, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(args.snapshot, use_fast=False)
    proc.tokenizer = tok
    proc.image_processor.min_pixels = args.min_pixels
    proc.image_processor.max_pixels = args.max_pixels

    s = json.load(open(args.sample))[args.idx]
    prompt = s["conversations"][0]["value"]
    # GT answer (assistant turn) for T2 token/trajectory exact-match
    gt_answer = s["conversations"][1]["value"] if len(s["conversations"]) > 1 else None
    # use the sample's own image paths directly (3 cameras in order) — robust to naming scheme
    images = [Image.open(os.path.join(args.img_dir, p)).convert("RGB") for p in s["image"]]
    msgs = messages_from_prompt(prompt, images)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=images, return_tensors="pt")
    prompt_ids = inputs.input_ids[0].cpu().numpy().astype(np.int64)
    pv = inputs.pixel_values.float().cpu().numpy()
    thw = inputs.image_grid_thw.cpu().numpy()

    # parity-validated scaffold + rope index (NO PyTorch model needed)
    x_t0, rbi, orig_len, _ = build_scaffold(prompt_ids, tok, args.snapshot)
    pos = get_rope_index_numpy(x_t0, thw)
    pos2d = pos[:, 0, :] if pos.ndim == 3 else pos

    out = {
        "input_ids": prompt_ids, "x_t0": x_t0, "rbi": rbi.astype(np.int64),
        "position_ids": pos2d.astype(np.int64), "pixel_values": pv.astype(np.float32),
        "image_grid_thw": thw.astype(np.int64), "orig_len": np.int64(orig_len),
    }
    if gt_answer is not None:
        out["target_ids"] = np.asarray(tok(gt_answer, add_special_tokens=False).input_ids, np.int64)

    if args.with_embeds:
        # Run the frozen ViT ONCE offline (fp32, highest precision) -> bf16 embeds reference.
        import ml_dtypes
        import jax, jax.numpy as jnp
        jax.config.update("jax_default_matmul_precision", "highest")
        from ddrive_jax.models.vision_qwen25vl import VisionConfig, VisionTransformer  # noqa: F401
        from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_vit
        vit, _ = load_fast_ddrive_vit(args.snapshot, VisionConfig(dtype=jnp.float32),
                                      dtype=jnp.float32, verbose=False)
        emb = np.asarray(vit(jnp.asarray(pv, jnp.float32), thw), np.float32)
        out["image_embeds"] = emb.astype(ml_dtypes.bfloat16)
        print(f"[prep] image_embeds {emb.shape} (fp32->bf16)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **out)
    print(f"[prep] saved {args.out}: L={x_t0.shape[0]} orig_len={orig_len} "
          f"image_tokens={int((prompt_ids == 151655).sum())} "
          f"{'+embeds' if args.with_embeds else ''} {'+target' if gt_answer else ''}")
    print("SASD_PREP_INPUTS_DONE")


if __name__ == "__main__":
    main()
