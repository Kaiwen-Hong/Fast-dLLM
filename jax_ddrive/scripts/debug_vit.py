"""Capture PyTorch ViT intermediates to isolate the JAX ViT parity gap."""
import sys, numpy as np, torch
import torch.nn.functional as F
SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
ORACLE = "/home/kaiwen/data/fast-ddrive/ref_logits/vit_oracle.npz"
OUT = "/home/kaiwen/data/fast-ddrive/ref_logits/vit_debug.npz"


def main():
    from transformers import AutoModelForCausalLM
    d = np.load(ORACLE)
    pv = torch.tensor(d["pixel_values"]).float().cuda()
    thw = torch.tensor(d["image_grid_thw"]).cuda()
    model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.float32, trust_remote_code=True,
                                                 attn_implementation="eager").eval().cuda()
    v = model.model.visual
    u = v.spatial_merge_unit
    with torch.no_grad():
        wi, cuw = v.get_window_index(thw)
        cuw = torch.tensor(cuw, dtype=torch.int32)
        cuw = torch.unique_consecutive(cuw)
        rpe = v.rot_pos_emb(thw)
        cu = torch.repeat_interleave(thw[:, 1] * thw[:, 2], thw[:, 0]).cumsum(0, dtype=torch.int32)
        cu = F.pad(cu, (1, 0), value=0)
        h = v.patch_embed(pv)
        sl = h.shape[0]
        h_r = h.reshape(sl // u, u, -1)[wi].reshape(sl, -1)
        rpe_r = rpe.reshape(sl // u, u, -1)[wi].reshape(sl, -1)
        emb = torch.cat([rpe_r, rpe_r], -1)
        cos, sin = emb.cos(), emb.sin()
        # block 0 output (window block)
        blk0 = v.blocks[0]
        pe = (cos, sin)
        b0 = blk0(h_r, cu_seqlens=cuw, position_embeddings=pe)
    np.savez(OUT, window_index=np.asarray(wi.cpu()), cu_window=np.asarray(cuw.cpu()),
             cu_seqlens=np.asarray(cu.cpu()), patch_reord=h_r.cpu().numpy(),
             cos=cos.cpu().numpy(), sin=sin.cpu().numpy(), block0=b0.cpu().numpy())
    print("saved", OUT, "| wi", wi.shape, "cuw", cuw.shape, "cu", cu.tolist()[:3], "...",
          "patch_reord", tuple(h_r.shape))
    print("VIT_DEBUG_DONE")


if __name__ == "__main__":
    main()
