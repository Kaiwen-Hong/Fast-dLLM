"""Phase-2 oracle: build the real SASD training batch for sample.json (text-only),
run the PyTorch forward + the model's OWN loss methods, and save every intermediate
so the JAX port can reproduce the loss exactly.

Determinism: a fixed (non-random) mask pattern, so JAX consumes identical noised inputs.
Forward done in EVAL mode over the doubled 2L seq with tiled positions [0..L-1,0..L-1]
(equivalent to training's q/k half-split, eager attention => additive mask).
"""
import json
import os
import sys

import numpy as np
import torch

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
SAMPLE = "/home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive/data/example/sample.json"
OUT = "/home/kaiwen/data/fast-ddrive/ref_logits/sasd_oracle.npz"
MASK_ID, IM_END, BD = 151665, 151645, 32
EXP_BUDGET = 192
sys.path.insert(0, SNAP)


def build_example(tokenizer):
    """Text-only (input_ids, labels) for sample.json[0], following the collator:
    explanation padded to 192 NULLs-budget, fmb values to 3, response-only labels,
    bd_size padding with MASK_ID."""
    import re
    s = json.load(open(SAMPLE))[0]
    conv = s["conversations"]
    human = conv[0]["value"].replace("<image>", "").strip()
    gpt = conv[1]["value"].replace("<|mdm_start|>", "").replace("<|mdm_end|>", "").replace("|<NULL>|", "<|NULL|>")
    obj = json.loads(gpt)
    # explanation NULL padding to a 32-multiple >= 192
    if "explanation" in obj:
        n = len(tokenizer.encode(obj["explanation"], add_special_tokens=False))
        pad = EXP_BUDGET - n if n < EXP_BUDGET else (BD - n % BD) % BD or BD
        obj["explanation"] = obj["explanation"] + "<|NULL|>" * pad
    if "future_meta_behavior" in obj:
        for k in ("longitudinal", "lateral"):
            if k in obj["future_meta_behavior"]:
                v = obj["future_meta_behavior"][k]
                p = 3 - len(tokenizer.encode(v, add_special_tokens=False))
                if p > 0:
                    obj["future_meta_behavior"][k] = v + "<|NULL|>" * p
    if isinstance(obj.get("trajectory"), str):
        t = re.sub(r'[+-]\d+\.\d+', lambda m: f"{m.group(0)[0]}{float(m.group(0)[1:]):06.2f}", obj["trajectory"])
        t = re.sub(r',([+-])', r', \1', t); t = re.sub(r'\[([+-])', r'[ \1', t)
        obj["trajectory"] = t
    gpt = json.dumps(obj, ensure_ascii=False)
    msgs = [{"role": "user", "content": human}, {"role": "assistant", "content": gpt}]
    ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    ids = list(ids)
    labels = [-100] * len(ids)
    # response region: scan [151644,77091,198] -> 151645
    for i in range(len(ids) - 2):
        if ids[i] == 151644 and ids[i + 1] == 77091 and ids[i + 2] == 198:
            rs = i + 3
            re_ = rs
            while re_ < len(ids):
                if ids[re_] == IM_END:
                    re_ += 1
                    break
                re_ += 1
            for j in range(rs, min(re_, len(ids))):
                labels[j] = ids[j]
            break
    # bd_size pad
    if len(ids) % BD:
        pad = BD - len(ids) % BD
        ids += [MASK_ID] * pad
        labels += [-100] * pad
    return np.array(ids, np.int64)[None], np.array(labels, np.int64)[None]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from section_utils import build_deep_scaffold_sequences, compute_section_block_idx_deep_static

    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.float32, trust_remote_code=True, attn_implementation="eager").eval().cuda()
    # grab the module-level mask fn from the dynamically-loaded modeling module
    hybrid_block_causal_mask_multiturn = sys.modules[type(model).__module__].hybrid_block_causal_mask_multiturn
    model._section_tokenizer = tok
    model.deep_json_scaffold = True
    V = model.config.text_config.vocab_size

    input_ids, labels = build_example(tok)
    L = input_ids.shape[1]
    print(f"L={L}  response tokens={(labels != -100).sum()}")
    ii = torch.tensor(input_ids); ll = torch.tensor(labels)

    seqs = build_deep_scaffold_sequences(tok)
    rbi, turn, n_blocks, scaffold_mask, block_to_section = compute_section_block_idx_deep_static(ll, ii, seqs, BD)
    rbi = rbi.view(-1).long(); turn = turn.view(-1).long(); scaffold_mask = scaffold_mask.view(-1).bool()
    print(f"n_blocks={n_blocks}  sections={block_to_section}")
    print(f"scaffold-frozen={int(scaffold_mask.sum())}  response={(ll[0] != -100).sum().item()}")

    # ---- deterministic noising (mirror modeling.py 2251-2346) ----
    resp = (ll != -100)
    idx = torch.arange(L)
    mask_indices = resp & (~scaffold_mask.unsqueeze(0)) & (idx % 2 == 0).unsqueeze(0)   # fixed pattern
    mask_indices = mask_indices | ((ii == IM_END) & resp)                                # always_mask_im_end
    noisy = ii.clone(); noisy[mask_indices] = MASK_ID
    lab_m = ll.clone(); lab_m[~mask_indices] = -100
    doubled = torch.cat([noisy, ii], dim=1)                                              # [1, 2L]
    # complementary
    comp = (resp & ~mask_indices) | ((ii == IM_END) & resp)
    comp = comp & (~scaffold_mask.unsqueeze(0))
    cnoisy = ii.clone(); cnoisy[comp] = MASK_ID
    clab = ll.clone(); clab[~comp] = -100
    cdoubled = torch.cat([cnoisy, ii], dim=1)
    input_final = torch.cat([doubled, cdoubled], dim=0)                                  # [2, 2L]
    labels_final = torch.cat([lab_m, clab], dim=0)                                       # [2, L]
    original_labels = ll.clone()                                                         # [1, L]

    weights = model._build_section_weight_tensor(labels_final, rbi, n_blocks, block_to_section=block_to_section)  # [2,L]

    # dense hybrid mask [2L,2L]
    qi = torch.arange(2 * L).view(2 * L, 1); ki = torch.arange(2 * L).view(1, 2 * L)
    dense = hybrid_block_causal_mask_multiturn(0, 0, qi, ki, response_block_idx=rbi, turn_idx=turn, n=L).bool()
    add_mask = torch.where(dense, 0.0, float("-inf")).view(1, 1, 2 * L, 2 * L).cuda()

    pos = torch.cat([torch.arange(L), torch.arange(L)]).view(1, 2 * L).cuda()
    num_items = float(2 * int((original_labels != -100).sum()))

    with torch.no_grad():
        emb = model.model.language_model.embed_tokens(input_final.cuda())               # [2,2L,D]
        hid = model.model.language_model(inputs_embeds=emb, attention_mask=add_mask.expand(2, 1, 2 * L, 2 * L),
                                         position_ids=pos.expand(2, 2 * L), use_cache=False).last_hidden_state
        noisy_logits = model.lm_head(hid[:, :L])                                         # [2,L,V]
        clean_logits = model.lm_head(hid[:1, L:])                                        # [1,L,V]
        primary = model.compute_section_weighted_loss(noisy_logits, labels_final.cuda(), weights.cuda(), V,
                                                      num_items_in_batch=num_items)
        comp_loss = model.loss_function(logits=clean_logits, labels=original_labels.cuda(), vocab_size=V,
                                        num_items_in_batch=num_items)
        total = primary + comp_loss

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT,
             input_final=input_final.numpy(), labels_final=labels_final.numpy(),
             original_labels=original_labels.numpy(), weights=weights.numpy(),
             rbi=rbi.numpy(), turn=turn.numpy(), dense_mask=dense.numpy(),
             position_ids=pos.cpu().numpy(), num_items=np.float64(num_items), L=np.int64(L),
             primary=float(primary), complementary=float(comp_loss), total=float(total),
             noisy_logits_slice=noisy_logits[:, :8, :64].float().cpu().numpy())
    print(f"\nTORCH loss  primary={float(primary):.6f}  complementary={float(comp_loss):.6f}  total={float(total):.6f}")
    print(f"saved {OUT}")
    print("ORACLE2_DONE")


if __name__ == "__main__":
    main()
