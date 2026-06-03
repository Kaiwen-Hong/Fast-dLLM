"""Capture the initial masked scaffold + structure + PyTorch reference output for the JAX
section-diffusion sampler. Text-only (pixel_values=None) for a clean parity (no ViT seed).
Monkeypatches the model's eval mask + forward to grab response_block_idx, x_t0, position_ids."""
import os, numpy as np, torch
SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
OUT = "/home/kaiwen/data/fast-ddrive/ref_logits/sd_inputs.npz"
PROMPT = ("You are an expert autonomous driving agent. Detect critical objects, explain the "
          "scene, predict meta-behavior, and produce a 5-second trajectory.")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.float32,
                                                 trust_remote_code=True).eval().cuda()
    msgs = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    prompt_len = ids.shape[1]

    cap = {}
    orig_mask = model.model.eval_hybrid_mask
    def wrap_mask(seqlen, rbi, *a, **k):
        if "rbi" not in cap and torch.is_tensor(rbi):
            cap["rbi"] = rbi.detach().cpu().numpy().astype(np.int32)
        return orig_mask(seqlen, rbi, *a, **k)
    model.model.eval_hybrid_mask = wrap_mask
    orig_fwd = model.forward
    def wrap_fwd(*a, **k):
        if "x_t0" not in cap:
            ii = k.get("input_ids", a[0] if a else None)
            if ii is not None and torch.is_tensor(ii):
                cap["x_t0"] = ii.detach().cpu().numpy().astype(np.int64)
            pid = k.get("position_ids")
            if pid is not None:
                cap["position_ids"] = pid.detach().cpu().numpy().astype(np.int32)
        return orig_fwd(*a, **k)
    model.forward = wrap_fwd

    with torch.no_grad():
        out = model.mdm_sample_deep_scaffold(ids, tok, pixel_values=None, image_grid_thw=None,
                                             threshold=0.9, block_size=32, max_tokens=512,
                                             use_kv_cache=False, temperature=0.0)
    out = out[0].detach().cpu().numpy().astype(np.int64)
    ref_gen = out[prompt_len:]
    print("x_t0", cap.get("x_t0", np.array([])).shape, "rbi", cap.get("rbi", np.array([])).shape,
          "pos", cap.get("position_ids", np.array([])).shape, "out", out.shape, "prompt_len", prompt_len)
    print("REF decoded:", tok.decode(ref_gen, skip_special_tokens=True)[:200])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, x_t0=cap["x_t0"], rbi=cap["rbi"], position_ids=cap["position_ids"],
             ref_output=out, prompt_len=np.int64(prompt_len))
    print("saved", OUT, "| SD_PREP_DONE")


if __name__ == "__main__":
    main()
