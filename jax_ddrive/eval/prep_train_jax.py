"""Prep a Fast-dDrive training JSON (with gpt targets) into per-sample npz for JAX SASD
training — generalises scripts/prep_overfit_data_mm.py to a dataset. PyTorch `ddrive` env,
CPU, NO model weights (uses section_utils + the validated numpy get_rope_index).

Each npz has the doubled-sequence SASD structures + frozen ViT inputs:
  input_ids[L], labels[L], rbi[L], turn[L], scaffold[L], weight_vec[L],
  block_alpha[n_blocks], block_beta[n_blocks], pixel_values[N,1176], image_grid_thw,
  position_ids[3,L], vision_mask[L].

Run: python jax_ddrive/eval/prep_train_jax.py --train_json .../train_targets.json \
        --image_root .../train_images --out_dir .../prep_train"""
import argparse, json, os, re, sys
import numpy as np

SNAP = os.environ.get("FASTDDRIVE_SNAP",
        "/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
sys.path.insert(0, os.environ.get("FASTDDRIVE_REPO", "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive"))
sys.path.insert(0, SNAP)

MASK_ID, IM_END, BD, EXP_BUDGET = 151665, 151645, 32, 32
IMAGE_TOK, VSTART = 151655, 151652
SECTION_W = {"critical_objects": 1.5, "explanation": 1.0, "future_meta_behavior": 2.0, "trajectory": 3.0}
NOISE_SCHED = {"critical_objects": (1.0, 2.0), "explanation": (1.0, 1.0),
               "future_meta_behavior": (1.0, 1.5), "trajectory": (2.0, 1.0)}


def process_gpt(gpt_raw, tok):
    """Replicate prep_overfit_data_mm.py gpt normalisation (strip mdm, pad NULL, fmt traj)."""
    gpt = (gpt_raw.replace("<|mdm_start|>", "").replace("<|mdm_end|>", "")
           .replace("|<NULL>|", "<|NULL|>"))
    obj = json.loads(gpt)
    if "explanation" in obj:
        n = len(tok.encode(obj["explanation"], add_special_tokens=False))
        pad = EXP_BUDGET - n if n < EXP_BUDGET else (BD - n % BD) % BD or BD
        obj["explanation"] += "<|NULL|>" * pad
    if "future_meta_behavior" in obj:
        for k in ("longitudinal", "lateral"):
            if k in obj["future_meta_behavior"]:
                v = obj["future_meta_behavior"][k]
                p = 3 - len(tok.encode(v, add_special_tokens=False))
                if p > 0:
                    obj["future_meta_behavior"][k] = v + "<|NULL|>" * p
    if isinstance(obj.get("trajectory"), str):
        t = re.sub(r'[+-]\d+\.\d+', lambda m: f"{m.group(0)[0]}{float(m.group(0)[1:]):06.2f}", obj["trajectory"])
        t = re.sub(r',([+-])', r', \1', t)
        t = re.sub(r'\[([+-])', r'[ \1', t)
        obj["trajectory"] = t
    return json.dumps(obj, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_pixels", type=int, default=784)
    ap.add_argument("--max_pixels", type=int, default=784 * 64)   # ~64 merged tokens/img
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoTokenizer
    from PIL import Image
    from section_utils import build_deep_scaffold_sequences, compute_section_block_idx_deep_static
    from ddrive_jax.eval.scaffold import messages_from_prompt
    from ddrive_jax.eval.rope_index import get_rope_index_numpy
    import torch

    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(SNAP, use_fast=False)
    proc.tokenizer = tok
    proc.image_processor.min_pixels = args.min_pixels
    proc.image_processor.max_pixels = args.max_pixels
    seqs = build_deep_scaffold_sequences(tok)

    data = json.load(open(args.train_json))
    if args.max_samples > 0:
        data = data[: args.max_samples]
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    for k, item in enumerate(data):
        human = item["conversations"][0]["value"]
        gpt = process_gpt(item["conversations"][1]["value"], tok)
        imgs = [Image.open(os.path.join(args.image_root, p)).convert("RGB") for p in item["image"]]
        user_content = messages_from_prompt(human, imgs)[0]["content"]
        msgs = [{"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": gpt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        inputs = proc(text=[text], images=imgs, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        pv = inputs["pixel_values"].float().cpu().numpy().astype(np.float16)
        thw = inputs["image_grid_thw"].cpu().numpy().astype(np.int64)
        if len(ids) % BD:
            ids += [MASK_ID] * (BD - len(ids) % BD)
        L = len(ids)
        labels = [-100] * L
        for i in range(L - 2):                       # response = after <|im_start|>assistant\n
            if ids[i] == 151644 and ids[i + 1] == 77091 and ids[i + 2] == 198:
                rs = i + 3; re_ = rs
                while re_ < L and ids[re_] != IM_END:
                    re_ += 1
                re_ = min(re_ + 1, L)
                for j in range(rs, re_):
                    labels[j] = ids[j]
                break
        ii = torch.tensor(ids)[None]; ll = torch.tensor(labels)[None]
        rbi, turn, n_blocks, scaff, b2s = compute_section_block_idx_deep_static(ll, ii, seqs, BD)
        rbi = rbi.view(-1).numpy().astype(np.int32); turn = turn.view(-1).numpy().astype(np.int32)
        scaff = scaff.view(-1).numpy().astype(bool)
        wv = np.ones(L, np.float32); ba = np.ones(n_blocks, np.float32); bb = np.ones(n_blocks, np.float32)
        for i, b in enumerate(rbi):
            if b >= 0 and int(b) in b2s and b2s[int(b)] in SECTION_W:
                wv[i] = SECTION_W[b2s[int(b)]]
        for b in range(n_blocks):
            if b2s.get(b) in NOISE_SCHED:
                ba[b], bb[b] = NOISE_SCHED[b2s[b]]
        pos = get_rope_index_numpy(np.array(ids, np.int64), thw).astype(np.int32)   # [3, L]
        vmask = np.array([(t == IMAGE_TOK) or (t == VSTART) or (t == 151654) for t in ids], bool)
        out = os.path.join(args.out_dir, f"{k:05d}.npz")
        np.savez(out, input_ids=np.array(ids, np.int64), labels=np.array(labels, np.int64),
                 rbi=rbi, turn=turn, scaffold=scaff, weight_vec=wv, block_alpha=ba, block_beta=bb,
                 n_blocks=np.int64(n_blocks), pixel_values=pv, image_grid_thw=thw,
                 position_ids=pos, vision_mask=vmask)
        manifest.append({"idx": k, "sample_id": item.get("sample_id", str(k)), "L": L,
                         "n_blocks": int(n_blocks), "npz": os.path.basename(out)})
        if (k + 1) % 50 == 0:
            print(f"  prepped {k+1}/{len(data)} (L={L}, blocks={n_blocks})", flush=True)
    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"))
    print(f"[prep-train] wrote {len(manifest)} npz → {args.out_dir}")
    print("PREP_TRAIN_JAX_DONE")


if __name__ == "__main__":
    main()
