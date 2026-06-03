"""Phase-4b: multimodal SASD overfit — show the FULL model (ViT image embeds fused into
the text stream + 3D M-RoPE + block-diffusion SASD loss) trains with decreasing loss.
ViT is frozen (image embeds computed once); the text decoder is fine-tuned (bf16+remat).
  python ddrive_jax/train_overfit_mm.py --steps 80 --lr 5e-5
"""
import argparse, sys, time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.models.qwen2_5_text import Qwen25TextConfig, mrope_cos_sin
from ddrive_jax.models.vision_qwen25vl import VisionConfig
from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_text, load_fast_ddrive_vit
from ddrive_jax.diffusion.masks import hybrid_block_causal_mask_dense, to_attn_mask4d
from ddrive_jax.diffusion.sasd_loss import section_weighted_ce, causal_ce
from ddrive_jax.diffusion import noise as noise_mod

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
DATA = "/home/kaiwen/data/fast-ddrive/ref_logits/overfit_data_mm.npz"
IMAGE_TOK = 151655


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()
    d = np.load(DATA)
    L = int(d["input_ids"].shape[0])
    s = {k: d[k] for k in ("input_ids", "labels", "rbi", "turn", "scaffold", "weight_vec", "block_alpha", "block_beta")}
    print(f"L={L} (doubled {2*L}); image tokens={int((d['input_ids']==IMAGE_TOK).sum())}")

    # fixed per-sample tensors
    mask4d = to_attn_mask4d(hybrid_block_causal_mask_dense(jnp.asarray(d["rbi"]), jnp.asarray(d["turn"]), L))
    pos_tiled = np.concatenate([d["position_ids"], d["position_ids"]], axis=1)        # [3, 2L]
    cosf, sinf = mrope_cos_sin(pos_tiled, 128, 1e6, (16, 24, 24))                     # [2L, 128]
    num_items = noise_mod.num_items(s)
    img_pos = jnp.asarray(np.where(np.tile(d["input_ids"], 2) == IMAGE_TOK)[0])       # [2*1365] in [0,2L)
    input_ids_2L = jnp.asarray(np.tile(d["input_ids"], 2))                            # [2L] (prompt repeats)

    print("loading text + ViT (bf16) ...", flush=True)
    text, _ = load_fast_ddrive_text(SNAP, Qwen25TextConfig.fast_ddrive(jnp.bfloat16), dtype=jnp.bfloat16, verbose=False)
    vit, _ = load_fast_ddrive_vit(SNAP, VisionConfig(dtype=jnp.bfloat16), dtype=jnp.bfloat16, verbose=False)
    image_embeds = jax.lax.stop_gradient(vit(jnp.asarray(d["pixel_values"], jnp.bfloat16), d["image_grid_thw"]))  # [1365, D]
    image_embeds = jnp.concatenate([image_embeds, image_embeds], 0).astype(jnp.bfloat16)  # [2*1365, D] for both halves
    del vit                                                                          # free the ViT
    print(f"image_embeds {image_embeds.shape} computed (frozen)")

    # fixed-mask batch (deterministic) for a clean monotonic curve
    resp = s["labels"] != -100
    fmask = resp & (~s["scaffold"]) & (np.arange(L) % 2 == 0)
    ifn, lfn, ol, w = noise_mod.make_batch(s, np.random.default_rng(0), fixed_mask=fmask)  # ifn [2,2L]
    ifn = jnp.asarray(ifn); lfn = jnp.asarray(lfn); ol = jnp.asarray(ol); w = jnp.asarray(w)

    def fuse(emb_2L):                          # [2L, D] text embeds -> scatter image embeds
        return emb_2L.at[img_pos].set(image_embeds.astype(emb_2L.dtype))

    def loss_fn(model, ifn, lfn, ol, w):
        # build doubled fused embeds [2, 2L, D]: each row = embed(noisy/comp half | clean half), vision restored
        rows = []
        for r in range(2):
            emb = fuse(model.embed_tokens(ifn[r]))                 # [2L, D]
            rows.append(emb)
        emb2 = jnp.stack(rows, 0)                                  # [2, 2L, D]
        hidden = model.hidden_forward_mrope_cs(emb2, cosf, sinf, mask4d, remat=True)
        noisy_logits = model.attend(hidden[:, :L, :])
        clean_logits = model.attend(hidden[:1, L:, :])
        return (section_weighted_ce(noisy_logits, lfn, w, num_items=num_items)
                + causal_ce(clean_logits, ol, num_items=num_items))

    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, max(args.steps // 10, 1),
                                               max(args.steps - args.steps // 10, 1), args.lr * 0.1)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adafactor(learning_rate=sched, multiply_by_parameter_scale=True, min_dim_size_to_factor=128))
    optimizer = nnx.Optimizer(text, tx, wrt=nnx.Param)

    @nnx.jit
    def step(model, optimizer, ifn, lfn, ol, w):
        l, g = nnx.value_and_grad(loss_fn)(model, ifn, lfn, ol, w)
        optimizer.update(model, g)
        return l

    @nnx.jit
    def evl(model, ifn, lfn, ol, w):
        return loss_fn(model, ifn, lfn, ol, w)

    l0 = float(evl(text, ifn, lfn, ol, w))
    print(f"[init] multimodal SASD loss = {l0:.4f}", flush=True)
    t0 = time.time()
    losses = [l0]
    for st in range(1, args.steps + 1):
        l = float(step(text, optimizer, ifn, lfn, ol, w))
        losses.append(l)
        if st % 10 == 0 or st == 1:
            print(f"step {st:3d} multimodal loss {l:.4f}  [{(time.time()-t0)/st*1000:.0f} ms/step]", flush=True)
    lf = losses[-1]
    print(f"\nINIT {l0:.4f} -> FINAL {lf:.4f}")
    print(f"PHASE4b_MM_TRAIN_{'PASS' if lf < l0 - 0.05 else 'FAIL'}: multimodal SASD loss {l0:.3f}->{lf:.3f}")
    sys.exit(0 if lf < l0 - 0.05 else 1)


if __name__ == "__main__":
    main()
