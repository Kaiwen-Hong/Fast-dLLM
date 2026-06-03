"""Capture a PyTorch section_diffusion (mdm_sample_deep_scaffold) MULTIMODAL reference for
one real sample, plus validate the JAX-side prep ports against the model's own internals:
  * build_scaffold(prompt_ids) == the x_t0 / response_block_idx the PyTorch sampler uses.
  * get_rope_index_numpy(x_t0, grid_thw) == model.model.get_rope_index(...).
Saves sd_mm_oracle.npz {input_ids(prompt), x_t0, rbi, position_ids, pixel_values,
image_grid_thw, orig_len, ref_output} for the JAX verify (scripts/verify_sd_mm.py).

Run (ddrive env, free GPU): python jax_ddrive/scripts/capture_oracle_sd_mm.py"""
import json, os, sys
import numpy as np
import torch

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
SAMPLE = "/home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive/data/example/sample.json"
IMG_DIR = "/home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive/data/example/images"
OUT = "/home/kaiwen/data/fast-ddrive/ref_logits/sd_mm_oracle.npz"
sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
sys.path.insert(0, SNAP)


def main():
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    from PIL import Image
    from ddrive_jax.eval.scaffold import build_scaffold, messages_from_prompt
    from ddrive_jax.eval.rope_index import get_rope_index_numpy

    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(SNAP, use_fast=False)
    proc.tokenizer = tok
    proc.image_processor.min_pixels = 200704   # paper eval resolution (batch_inference)
    proc.image_processor.max_pixels = 200704
    model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.float32,
                                                 trust_remote_code=True).eval().cuda()

    s = json.load(open(SAMPLE))[0]
    prompt = s["conversations"][0]["value"]
    fid = os.path.basename(s["image"][1]).split("_")[0]   # "161"
    images = [Image.open(f"{IMG_DIR}/{fid}_CAM_FRONT_LEFT.jpg").convert("RGB"),
              Image.open(f"{IMG_DIR}/{fid}_CAM_FRONT.jpg").convert("RGB"),
              Image.open(f"{IMG_DIR}/{fid}_CAM_FRONT_RIGHT.jpg").convert("RGB")]
    msgs = messages_from_prompt(prompt, images)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=images, return_tensors="pt").to("cuda")
    prompt_ids = inputs.input_ids[0].cpu().numpy().astype(np.int64)
    pv = inputs.pixel_values.float().cpu().numpy()
    thw = inputs.image_grid_thw.cpu().numpy()
    print(f"prompt_len={len(prompt_ids)} pixel_values={pv.shape} grid_thw={thw.tolist()} "
          f"image_tokens={int((prompt_ids==151655).sum())}")

    cap = {}
    orig_mask = model.model.eval_hybrid_mask
    def wrap_mask(seqlen, rbi, *a, **k):
        if "rbi" not in cap and torch.is_tensor(rbi):
            cap["rbi"] = rbi.detach().cpu().numpy().astype(np.int64)
        return orig_mask(seqlen, rbi, *a, **k)
    model.model.eval_hybrid_mask = wrap_mask
    orig_fwd = model.forward
    def wrap_fwd(*a, **k):
        if "x_t0" not in cap:
            ii = k.get("input_ids", a[0] if a else None)
            if torch.is_tensor(ii):
                cap["x_t0"] = ii.detach().cpu().numpy().astype(np.int64)[0]
            pid = k.get("position_ids")
            if torch.is_tensor(pid):
                cap["position_ids"] = pid.detach().cpu().numpy().astype(np.int64)
        return orig_fwd(*a, **k)
    model.forward = wrap_fwd

    with torch.no_grad():
        out = model.mdm_sample_deep_scaffold(
            inputs.input_ids, tok, pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw, threshold=0.9, block_size=32,
            max_tokens=512, use_kv_cache=False, temperature=0.0)
    out = out[0].detach().cpu().numpy().astype(np.int64)
    model.forward = orig_fwd
    x_t0 = cap["x_t0"]; rbi = cap["rbi"].reshape(-1); pos = cap["position_ids"]
    pos2d = pos[:, 0, :] if pos.ndim == 3 else pos
    orig_len = len(prompt_ids)
    print("REF section_diffusion:", tok.decode(out[orig_len:], skip_special_tokens=True)[:200])

    # ── validate the JAX prep ports against PyTorch internals ──
    my_x_t0, my_rbi, my_orig, _ = build_scaffold(prompt_ids, tok, SNAP)
    ok_xt0 = my_x_t0.shape == x_t0.shape and bool((my_x_t0 == x_t0).all())
    ok_rbi = my_rbi.shape == rbi.shape and bool((my_rbi == rbi).all())
    my_pos = get_rope_index_numpy(x_t0, thw)
    ok_pos = my_pos.shape == pos2d.shape and bool((my_pos == pos2d).all())
    print(f"[validate] build_scaffold x_t0 match: {ok_xt0} | rbi match: {ok_rbi} | "
          f"numpy get_rope_index match: {ok_pos}")
    if not ok_rbi:
        d = np.where(my_rbi != rbi)[0]
        print("  rbi first diffs:", d[:10], "mine", my_rbi[d[:10]], "ref", rbi[d[:10]])
    if not ok_pos:
        d = np.where(my_pos != pos2d)
        print("  pos diffs at", list(zip(*[x[:8] for x in d])))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, input_ids=prompt_ids, x_t0=x_t0, rbi=rbi, position_ids=pos2d,
             pixel_values=pv, image_grid_thw=thw, orig_len=np.int64(orig_len), ref_output=out)
    print("saved", OUT)
    print("SD_MM_PREP_PORTS_" + ("PASS" if (ok_xt0 and ok_rbi and ok_pos) else "FAIL"))


if __name__ == "__main__":
    main()
