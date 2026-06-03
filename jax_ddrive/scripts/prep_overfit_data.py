"""Prep the 2 example samples into a JAX-trainable dataset (tokenizer + section_utils only;
no 16GB model load). For each sample saves: input_ids[L], labels[L], rbi[L], turn[L],
scaffold_mask[L], weight_vec[L] (section loss weights), and the block->section list.
Prompt is left-truncated to keep L small for the 5090 memory budget.
"""
import json
import os
import re
import sys

import numpy as np

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
SAMPLE = "/home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive/data/example/sample.json"
OUT = "/home/kaiwen/data/fast-ddrive/ref_logits/overfit_data.npz"
MASK_ID, IM_END, BD = 151665, 151645, 32
EXP_BUDGET = 32
HUMAN_CHARS = 256           # truncate the long task prompt for memory (shared 5090)
SECTION_W = {"critical_objects": 1.5, "explanation": 1.0, "future_meta_behavior": 2.0, "trajectory": 3.0}
NOISE_SCHED = {"critical_objects": (1.0, 2.0), "explanation": (1.0, 1.0),
               "future_meta_behavior": (1.0, 1.5), "trajectory": (2.0, 1.0)}
sys.path.insert(0, SNAP)


def build(sample, tok, seqs, csbi):
    human = sample["conversations"][0]["value"].replace("<image>", "").strip()[:HUMAN_CHARS]
    gpt = sample["conversations"][1]["value"].replace("<|mdm_start|>", "").replace("<|mdm_end|>", "").replace("|<NULL>|", "<|NULL|>")
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
        t = re.sub(r',([+-])', r', \1', t); t = re.sub(r'\[([+-])', r'[ \1', t)
        obj["trajectory"] = t
    gpt = json.dumps(obj, ensure_ascii=False)
    ids = list(tok.apply_chat_template([{"role": "user", "content": human},
                                        {"role": "assistant", "content": gpt}],
                                       tokenize=True, add_generation_prompt=False))
    labels = [-100] * len(ids)
    for i in range(len(ids) - 2):
        if ids[i] == 151644 and ids[i + 1] == 77091 and ids[i + 2] == 198:
            rs = i + 3; re_ = rs
            while re_ < len(ids):
                if ids[re_] == IM_END:
                    re_ += 1; break
                re_ += 1
            for j in range(rs, min(re_, len(ids))):
                labels[j] = ids[j]
            break
    if len(ids) % BD:
        pad = BD - len(ids) % BD
        ids += [MASK_ID] * pad; labels += [-100] * pad
    import torch
    ii = torch.tensor(ids)[None]; ll = torch.tensor(labels)[None]
    rbi, turn, n_blocks, scaff, b2s = csbi(ll, ii, seqs, BD)
    rbi = rbi.view(-1).numpy().astype(np.int32); turn = turn.view(-1).numpy().astype(np.int32)
    scaff = scaff.view(-1).numpy().astype(bool)
    wv = np.ones(len(ids), np.float32)
    for i, b in enumerate(rbi):
        if b >= 0 and int(b) in b2s and b2s[int(b)] in SECTION_W:
            wv[i] = SECTION_W[b2s[int(b)]]
    block_a = np.ones(n_blocks, np.float32); block_b = np.ones(n_blocks, np.float32)
    for b in range(n_blocks):
        sec = b2s.get(b)
        if sec in NOISE_SCHED:
            block_a[b], block_b[b] = NOISE_SCHED[sec]
    return dict(input_ids=np.array(ids, np.int64), labels=np.array(labels, np.int64),
                rbi=rbi, turn=turn, scaffold=scaff, weight_vec=wv, n_blocks=int(n_blocks),
                block_alpha=block_a, block_beta=block_b)


def main():
    from transformers import AutoTokenizer
    from section_utils import build_deep_scaffold_sequences, compute_section_block_idx_deep_static
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    seqs = build_deep_scaffold_sequences(tok)
    data = json.load(open(SAMPLE))
    out = {}
    for si, s in enumerate(data):
        d = build(s, tok, seqs, compute_section_block_idx_deep_static)
        L = len(d["input_ids"]); resp = int((d["labels"] != -100).sum())
        print(f"sample {si}: L={L} response={resp} n_blocks={d['n_blocks']} scaffold={int(d['scaffold'].sum())}")
        for k, v in d.items():
            out[f"s{si}_{k}"] = v
    out["n_samples"] = np.int64(len(data))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **out)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
