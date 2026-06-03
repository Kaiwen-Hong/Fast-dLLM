"""Fast-dDrive JAX SASD training on REAL Waymo data (multi-sample, the production training
loop — generalises train_overfit_mm.py). Each step: pick a real sample, stochastically
noise its doubled [noisy|clean] scaffold (per-section Beta schedule), and minimise the
Section-Importance-Weighted Loss + complementary-mask causal loss. ViT image embeds are
precomputed once per sample (frozen). Orbax checkpoints periodically.

Canonical recipe (train_waymo_sasd.sh): section weights {CO 1.5, exp 1.0, FMB 2.0, traj 3.0},
per-section Beta noise, bd_size 32, deep JSON scaffold. bf16 + per-layer remat + Adafactor.

Run (jax env, free GPU):
  XLA_PYTHON_CLIENT_PREALLOCATE=false python jax_ddrive/ddrive_jax/train_waymo_sasd_jax.py \
      --prep_dir .../prep_train --samples 200 --steps 400 --lr 2e-5 --ckpt_dir .../ckpt_jax"""
import argparse, json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
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
IMAGE_TOK = 151655


def load_sample(npz_path):
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}


def static_tensors(s, vit, dtype):
    """Per-sample fixed tensors for the doubled SASD forward (+ frozen ViT embeds)."""
    L = int(s["input_ids"].shape[0])
    mask4d = to_attn_mask4d(hybrid_block_causal_mask_dense(
        jnp.asarray(s["rbi"]), jnp.asarray(s["turn"]), L))
    pos_tiled = np.concatenate([s["position_ids"], s["position_ids"]], axis=1)
    cosf, sinf = mrope_cos_sin(pos_tiled, 128, 1e6, (16, 24, 24))
    img_pos = jnp.asarray(np.where(np.tile(s["input_ids"], 2) == IMAGE_TOK)[0])
    ie = jax.lax.stop_gradient(vit(jnp.asarray(s["pixel_values"], dtype), s["image_grid_thw"]))
    ie = jnp.concatenate([ie, ie], 0).astype(dtype)
    num_items = float(noise_mod.num_items(s))
    return dict(L=L, mask4d=mask4d, cosf=cosf, sinf=sinf, img_pos=img_pos,
                image_embeds=ie, num_items=jnp.float32(num_items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_dir", required=True)
    ap.add_argument("--samples", type=int, default=200, help="how many prepped samples to train on")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--ckpt_dir", default="/home/kaiwen/data/fast-ddrive/ckpt/jax_waymo_sasd")
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.prep_dir, "manifest.json")))[: args.samples]
    print(f"training on {len(manifest)} real Waymo samples (L set: "
          f"{sorted(set(m['L'] for m in manifest))[:6]}...)", flush=True)

    text, _ = load_fast_ddrive_text(SNAP, Qwen25TextConfig.fast_ddrive(jnp.bfloat16),
                                    dtype=jnp.bfloat16, verbose=False)
    vit, _ = load_fast_ddrive_vit(SNAP, VisionConfig(dtype=jnp.bfloat16),
                                  dtype=jnp.bfloat16, verbose=False)

    # precompute per-sample static tensors (incl. frozen ViT embeds), then free the ViT
    samples = []
    print("precomputing ViT image embeds + static tensors ...", flush=True)
    for m in manifest:
        s = load_sample(os.path.join(args.prep_dir, m["npz"]))
        samples.append((s, static_tensors(s, vit, jnp.bfloat16)))
    del vit
    print(f"precomputed {len(samples)} samples", flush=True)

    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, max(args.steps // 10, 1),
                                               max(args.steps - args.steps // 10, 1), args.lr * 0.1)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adafactor(learning_rate=sched, multiply_by_parameter_scale=True,
                                     min_dim_size_to_factor=128))
    optimizer = nnx.Optimizer(text, tx, wrt=nnx.Param)

    @nnx.jit
    def step(model, optimizer, ifn, lfn, ol, w, cosf, sinf, mask4d, img_pos, image_embeds, num_items):
        Lh = ifn.shape[1] // 2

        def loss_fn(model):
            rows = []
            for r in range(2):
                emb = model.embed_tokens(ifn[r]).at[img_pos].set(image_embeds.astype(model.embed_tokens(ifn[r]).dtype))
                rows.append(emb)
            emb2 = jnp.stack(rows, 0)
            hidden = model.hidden_forward_mrope_cs(emb2, cosf, sinf, mask4d, remat=True)
            nl = model.attend(hidden[:, :Lh, :])
            cl = model.attend(hidden[:1, Lh:, :])
            return (section_weighted_ce(nl, lfn, w, num_items=num_items)
                    + causal_ce(cl, ol, num_items=num_items))

        l, g = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, g)
        return l

    @nnx.jit
    def evl(model, ifn, lfn, ol, w, cosf, sinf, mask4d, img_pos, image_embeds, num_items):
        Lh = ifn.shape[1] // 2
        rows = []
        for r in range(2):
            emb = model.embed_tokens(ifn[r]).at[img_pos].set(image_embeds.astype(model.embed_tokens(ifn[r]).dtype))
            rows.append(emb)
        emb2 = jnp.stack(rows, 0)
        hidden = model.hidden_forward_mrope_cs(emb2, cosf, sinf, mask4d)
        nl = model.attend(hidden[:, :Lh, :]); cl = model.attend(hidden[:1, Lh:, :])
        return (section_weighted_ce(nl, lfn, w, num_items=num_items)
                + causal_ce(cl, ol, num_items=num_items))

    # fixed eval batch (sample 0, deterministic noise) for a clean monotonic curve
    s0, t0s = samples[0]
    L0 = t0s["L"]
    resp0 = s0["labels"] != -100
    fmask0 = resp0 & (~s0["scaffold"]) & (np.arange(L0) % 2 == 0)
    eb = noise_mod.make_batch(s0, np.random.default_rng(123), fixed_mask=fmask0)
    eb = [jnp.asarray(x) for x in eb]

    def ev():
        return float(evl(text, eb[0], eb[1], eb[2], eb[3], t0s["cosf"], t0s["sinf"],
                         t0s["mask4d"], t0s["img_pos"], t0s["image_embeds"], t0s["num_items"]))

    rng = np.random.default_rng(args.seed)
    l_init = ev()
    print(f"[init] fixed-eval SASD loss = {l_init:.4f}", flush=True)
    t0 = time.time()
    run = []
    for st in range(1, args.steps + 1):
        s, ts = samples[rng.integers(len(samples))]
        ifn, lfn, ol, w = noise_mod.make_batch(s, rng)
        l = float(step(text, optimizer, jnp.asarray(ifn), jnp.asarray(lfn), jnp.asarray(ol),
                       jnp.asarray(w), ts["cosf"], ts["sinf"], ts["mask4d"], ts["img_pos"],
                       ts["image_embeds"], ts["num_items"]))
        run.append(l)
        if st % 20 == 0 or st == 1:
            print(f"step {st:4d} train {l:.4f} | eval {ev():.4f} | "
                  f"[{(time.time()-t0)/st*1000:.0f} ms/step]", flush=True)
        if args.save_every and st % args.save_every == 0:
            save_ckpt(text, args.ckpt_dir, st)
    l_final = ev()
    save_ckpt(text, args.ckpt_dir, args.steps)
    print(f"\n[done] fixed-eval SASD loss {l_init:.4f} -> {l_final:.4f} "
          f"(train last-20 mean {np.mean(run[-20:]):.4f})")
    ok = l_final < l_init - 0.05
    print(f"WAYMO_SASD_JAX_TRAIN_{'PASS' if ok else 'FAIL'}: real-data loss {l_init:.3f}->{l_final:.3f}")
    sys.exit(0 if ok else 1)


def save_ckpt(model, ckpt_dir, step):
    try:
        import orbax.checkpoint as ocp
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(ckpt_dir, f"step_{step}"))
        ckptr = ocp.StandardCheckpointer()
        ckptr.save(path, nnx.state(model), force=True)
        ckptr.wait_until_finished()
        print(f"  [ckpt] saved → {path}", flush=True)
    except Exception as e:
        print(f"  [ckpt] save failed: {type(e).__name__}: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()
