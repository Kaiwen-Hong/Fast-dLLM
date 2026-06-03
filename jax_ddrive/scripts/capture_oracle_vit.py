"""Phase-4 oracle: run the PyTorch Qwen2.5-VL vision tower on one example image and
save (pixel_values, image_grid_thw, image_embeds) for the JAX ViT parity gate.
"""
import os
import numpy as np
import torch
from PIL import Image

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
IMG = "/home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive/data/example/images/161_CAM_FRONT.jpg"
OUT = "/home/kaiwen/data/fast-ddrive/ref_logits/vit_oracle.npz"


def main():
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(SNAP, use_fast=False)
    proc.tokenizer = tok
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.float32, trust_remote_code=True, attn_implementation="eager").eval().cuda()

    img = Image.open(IMG).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "Describe."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt")
    pv = inputs["pixel_values"].float()
    thw = inputs["image_grid_thw"]
    print("pixel_values", tuple(pv.shape), "image_grid_thw", thw.tolist())

    with torch.no_grad():
        emb = model.model.visual(pv.cuda(), grid_thw=thw.cuda())     # [N//merge^2, out_hidden]
    print("image_embeds", tuple(emb.shape), "abs-max", float(emb.abs().max()))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, pixel_values=pv.cpu().numpy(), image_grid_thw=thw.cpu().numpy(),
             image_embeds=emb.float().cpu().numpy())
    print(f"saved {OUT}")
    print("VIT_ORACLE_DONE")


if __name__ == "__main__":
    main()
