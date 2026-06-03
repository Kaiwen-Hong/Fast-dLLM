"""Prep ONE image-containing sample for multimodal SASD overfit. Saves input_ids, labels,
rbi/turn/scaffold/weight_vec/block_alpha,beta (section structure), pixel_values,
image_grid_thw, 3D position_ids (get_rope_index), and vision_token_mask.
Loads the model once only for the authoritative get_rope_index + processor."""
import json, os, re, sys
import numpy as np, torch
from PIL import Image
SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
SAMPLE = "/home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive/data/example/sample.json"
IMG = "/home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive/data/example/images/161_CAM_FRONT.jpg"
OUT = "/home/kaiwen/data/fast-ddrive/ref_logits/overfit_data_mm.npz"
MASK_ID, IM_END, BD, EXP_BUDGET = 151665, 151645, 32, 32
IMAGE_TOK, VSTART = 151655, 151652
SECTION_W = {"critical_objects": 1.5, "explanation": 1.0, "future_meta_behavior": 2.0, "trajectory": 3.0}
NOISE_SCHED = {"critical_objects": (1.0, 2.0), "explanation": (1.0, 1.0),
               "future_meta_behavior": (1.0, 1.5), "trajectory": (2.0, 1.0)}
sys.path.insert(0, SNAP)


def main():
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    from section_utils import build_deep_scaffold_sequences, compute_section_block_idx_deep_static
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(SNAP, use_fast=False); proc.tokenizer = tok
    # shrink the image so the doubled training seq fits the shared 5090 (full-FT).
    proc.image_processor.min_pixels = 784
    proc.image_processor.max_pixels = 784 * 100   # ~100 merged image tokens
    model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.float32, trust_remote_code=True,
                                                 attn_implementation="eager").eval().cuda()

    s = json.load(open(SAMPLE))[0]
    human = s["conversations"][0]["value"].replace("<image>", "").strip()[:300]
    gpt = s["conversations"][1]["value"].replace("<|mdm_start|>", "").replace("<|mdm_end|>", "").replace("|<NULL>|", "<|NULL|>")
    obj = json.loads(gpt)
    if "explanation" in obj:
        n = len(tok.encode(obj["explanation"], add_special_tokens=False))
        pad = EXP_BUDGET - n if n < EXP_BUDGET else (BD - n % BD) % BD or BD
        obj["explanation"] += "<|NULL|>" * pad
    if "future_meta_behavior" in obj:
        for k in ("longitudinal", "lateral"):
            if k in obj["future_meta_behavior"]:
                v = obj["future_meta_behavior"][k]; p = 3 - len(tok.encode(v, add_special_tokens=False))
                if p > 0: obj["future_meta_behavior"][k] = v + "<|NULL|>" * p
    if isinstance(obj.get("trajectory"), str):
        t = re.sub(r'[+-]\d+\.\d+', lambda m: f"{m.group(0)[0]}{float(m.group(0)[1:]):06.2f}", obj["trajectory"])
        t = re.sub(r',([+-])', r', \1', t); t = re.sub(r'\[([+-])', r'[ \1', t); obj["trajectory"] = t
    gpt = json.dumps(obj, ensure_ascii=False)

    img = Image.open(IMG).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": human}]},
            {"role": "assistant", "content": [{"type": "text", "text": gpt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    inputs = proc(text=[text], images=[img], return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    pv = inputs["pixel_values"].float().cpu().numpy(); thw = inputs["image_grid_thw"].cpu().numpy()
    # bd pad
    if len(ids) % BD:
        ids += [MASK_ID] * (BD - len(ids) % BD)
    L = len(ids)
    labels = [-100] * L
    for i in range(L - 2):
        if ids[i] == 151644 and ids[i + 1] == 77091 and ids[i + 2] == 198:
            rs = i + 3; re_ = rs
            while re_ < L and ids[re_] != IM_END: re_ += 1
            re_ = min(re_ + 1, L)
            for j in range(rs, re_): labels[j] = ids[j]
            break
    ii = torch.tensor(ids)[None]; ll = torch.tensor(labels)[None]
    seqs = build_deep_scaffold_sequences(tok)
    rbi, turn, n_blocks, scaff, b2s = compute_section_block_idx_deep_static(ll, ii, seqs, BD)
    rbi = rbi.view(-1).numpy().astype(np.int32); turn = turn.view(-1).numpy().astype(np.int32)
    scaff = scaff.view(-1).numpy().astype(bool)
    wv = np.ones(L, np.float32); ba = np.ones(n_blocks, np.float32); bb = np.ones(n_blocks, np.float32)
    for i, b in enumerate(rbi):
        if b >= 0 and int(b) in b2s and b2s[int(b)] in SECTION_W: wv[i] = SECTION_W[b2s[int(b)]]
    for b in range(n_blocks):
        if b2s.get(b) in NOISE_SCHED: ba[b], bb[b] = NOISE_SCHED[b2s[b]]
    with torch.no_grad():
        pos, _ = model.model.get_rope_index(ii.cuda(), torch.tensor(thw).cuda(), None, attention_mask=None)
    pos = pos[:, 0, :].cpu().numpy().astype(np.int32)              # [3, L]
    vmask = np.array([(t == IMAGE_TOK) or (t == VSTART) or (t == 151654) for t in ids], bool)
    print(f"L={L} response={(np.array(labels)!=-100).sum()} n_blocks={n_blocks} image_tokens={int((np.array(ids)==IMAGE_TOK).sum())} sections={b2s}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, input_ids=np.array(ids, np.int64), labels=np.array(labels, np.int64),
             rbi=rbi, turn=turn, scaffold=scaff, weight_vec=wv, block_alpha=ba, block_beta=bb,
             n_blocks=np.int64(n_blocks), pixel_values=pv, image_grid_thw=thw,
             position_ids=pos, vision_mask=vmask)
    print("saved", OUT)


if __name__ == "__main__":
    main()
