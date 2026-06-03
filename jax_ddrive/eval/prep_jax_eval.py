"""Prep a Fast-dDrive EVAL_JSON for JAX inference (PyTorch `ddrive` env; CPU, NO model
weights — only the HF processor + section_utils scaffold layout). For each sample writes
an npz with everything the JAX neural compute needs:
  x_t0 [L], rbi [L], position_ids [3,L], pixel_values [N,1176] (fp16), image_grid_thw,
  orig_len, sample_id, future_waypoints (GT passthrough for predictions.json).

Run: python jax_ddrive/eval/prep_jax_eval.py --eval_json val_rated.json \
        --image_root .../val_images --out_dir .../prep_val --min_pixels 200704"""
import argparse, json, os, sys
import numpy as np
from PIL import Image

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
sys.path.insert(0, SNAP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_pixels", type=int, default=200704)
    ap.add_argument("--max_pixels", type=int, default=200704)
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoTokenizer
    from ddrive_jax.eval.scaffold import build_scaffold, messages_from_prompt
    from ddrive_jax.eval.rope_index import get_rope_index_numpy

    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(SNAP, use_fast=False)
    proc.tokenizer = tok
    proc.image_processor.min_pixels = args.min_pixels
    proc.image_processor.max_pixels = args.max_pixels

    data = json.load(open(args.eval_json))
    if args.start:
        data = data[args.start:]
    if args.max_samples > 0:
        data = data[: args.max_samples]
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    for k, item in enumerate(data):
        sid = item["sample_id"]
        prompt = item["conversations"][0]["value"]
        imgs = [Image.open(os.path.join(args.image_root, p)).convert("RGB") for p in item["image"]]
        msgs = messages_from_prompt(prompt, imgs)
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=imgs, return_tensors="pt")
        prompt_ids = inputs.input_ids[0].cpu().numpy().astype(np.int64)
        pv = inputs.pixel_values.float().cpu().numpy().astype(np.float16)
        thw = inputs.image_grid_thw.cpu().numpy().astype(np.int64)
        x_t0, rbi, orig_len, _ = build_scaffold(prompt_ids, tok, SNAP)
        pos = get_rope_index_numpy(x_t0, thw)
        fw = np.asarray(item.get("future waypoints", []), np.float32)
        out = os.path.join(args.out_dir, f"{args.start + k:05d}.npz")
        np.savez(out, sample_id=sid, x_t0=x_t0, rbi=rbi, position_ids=pos,
                 pixel_values=pv, image_grid_thw=thw, orig_len=np.int64(orig_len),
                 future_waypoints=fw)
        manifest.append({"idx": args.start + k, "sample_id": sid, "L": int(x_t0.shape[0]),
                         "npz": os.path.basename(out)})
        if (k + 1) % 25 == 0:
            print(f"  prepped {k+1}/{len(data)} (L={x_t0.shape[0]})", flush=True)
    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"))
    print(f"[prep] wrote {len(manifest)} npz → {args.out_dir}")
    print("PREP_JAX_EVAL_DONE")


if __name__ == "__main__":
    main()
