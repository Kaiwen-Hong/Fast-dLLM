"""Phase-4b oracle: full multimodal forward. Builds (image+prompt) inputs, fuses ViT
embeds into the text stream, computes 3D M-RoPE positions, runs the text decoder with a
bidirectional mask, and saves the FUSED inputs_embeds + 3D position_ids + hidden + logits.
The JAX side reproduces from the SAME fused embeds (isolating M-RoPE+decoder from the ViT
cuDNN seed)."""
import os, numpy as np, torch
from PIL import Image
SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
IMG = "/home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive/data/example/images/161_CAM_FRONT.jpg"
OUT = "/home/kaiwen/data/fast-ddrive/ref_logits/mm_oracle.npz"


def main():
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(SNAP, use_fast=False); proc.tokenizer = tok
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.float32, trust_remote_code=True, attn_implementation="eager").eval().cuda()
    M = model.model
    img = Image.open(IMG).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "Describe the scene."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    ids = inputs["input_ids"]; pv = inputs["pixel_values"].float(); thw = inputs["image_grid_thw"]
    L = ids.shape[1]
    print("input_ids", tuple(ids.shape), "n image tokens", int((ids == 151655).sum()))

    with torch.no_grad():
        emb = M.get_input_embeddings()(ids)
        img_emb = M.get_image_features(pv, thw)
        img_emb = torch.cat(img_emb, 0).to(emb.dtype)
        mask_img, _ = M.get_placeholder_mask(ids, inputs_embeds=emb, image_features=img_emb)
        fused = emb.masked_scatter(mask_img, img_emb)                       # [1, L, D]
        pos, _ = M.get_rope_index(ids, thw, None, attention_mask=inputs.get("attention_mask"))  # [3,1,L]
        add_mask = torch.zeros(1, 1, L, L, device="cuda")                   # bidirectional
        hid = M.language_model(inputs_embeds=fused, position_ids=pos, attention_mask=add_mask,
                               use_cache=False).last_hidden_state
        logits = model.lm_head(hid)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, input_ids=ids.cpu().numpy(), fused_embeds=fused.float().cpu().numpy(),
             position_ids=pos.cpu().numpy(), hidden=hid.float().cpu().numpy(),
             logits=logits.float().cpu().numpy())
    print("position_ids", tuple(pos.shape), "fused", tuple(fused.shape), "logits", tuple(logits.shape))
    print("saved", OUT, "| MM_ORACLE_DONE")


if __name__ == "__main__":
    main()
