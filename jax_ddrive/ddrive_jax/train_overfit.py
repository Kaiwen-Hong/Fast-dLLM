"""Phase-3 milestone: overfit the 2 example samples and show SASD training loss decreases.

bf16 params + Adafactor (factored state) to fit the 5090 alongside other GPU users.
Reuses the Phase-2-validated forward + loss. Run:
  python -m ddrive_jax.train_overfit --source trained --steps 150 --lr 1e-4
"""
import argparse
import csv
import sys
import time

import numpy as np
import jax
# Training uses default (TF32) matmul precision — faster + lower memory than the
# 'highest' fp32 used in the parity scripts. Loss still decreases identically.
import jax.numpy as jnp
import optax
from flax import nnx

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.models.qwen2_5_text import Qwen25TextConfig
from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_text
from ddrive_jax.diffusion.masks import hybrid_block_causal_mask_dense, to_attn_mask4d
from ddrive_jax.diffusion.sasd_loss import section_weighted_ce, causal_ce
from ddrive_jax.diffusion import noise as noise_mod
from ddrive_jax.lora import LoRAConfig, apply_lora, freeze_non_lora_params

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
DATA = "/home/kaiwen/data/fast-ddrive/ref_logits/overfit_data.npz"
CSV = "/home/kaiwen/data/fast-ddrive/logs/overfit_loss_curve.csv"


def loss_fn(model, input_final, mask4d, posB, labels_final, weights, original_labels, num_items):
    hidden = model.hidden_forward(input_final, mask4d, posB, remat=True)   # [2, 2L, D] bf16, ckpt
    half = hidden.shape[1] // 2
    noisy_logits = model.attend(hidden[:, :half, :])               # [2, L, V]
    clean_logits = model.attend(hidden[:1, half:, :])              # [1, L, V]
    primary = section_weighted_ce(noisy_logits, labels_final, weights, num_items=num_items)
    comp = causal_ce(clean_logits, original_labels, num_items=num_items)
    return primary + comp


@nnx.jit
def train_step(model, optimizer, input_final, mask4d, posB, labels_final, weights, original_labels, num_items):
    l, grads = nnx.value_and_grad(loss_fn)(model, input_final, mask4d, posB, labels_final, weights,
                                           original_labels, num_items)
    optimizer.update(model, grads)
    return l


@nnx.jit
def eval_loss(model, input_final, mask4d, posB, labels_final, weights, original_labels, num_items):
    return loss_fn(model, input_final, mask4d, posB, labels_final, weights, original_labels, num_items)


def load_samples():
    d = np.load(DATA)
    n = int(d["n_samples"])
    samples = []
    for i in range(n):
        s = {k[len(f"s{i}_"):]: d[k] for k in d.files if k.startswith(f"s{i}_")}
        samples.append(s)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="trained", choices=["trained", "base"])
    ap.add_argument("--base_snap", default=None, help="snapshot dir for Qwen2.5-VL-3B base")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full_ft", action="store_true", help="full fine-tune (OOMs on shared 5090); default = LoRA")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--fixed_batch", action="store_true",
                    help="train on the SAME fixed-mask batch every step (clean monotonic overfit curve)")
    args = ap.parse_args()

    samples = load_samples()
    print(f"{len(samples)} samples; L = {[int(s['input_ids'].shape[0]) for s in samples]}")

    # per-sample fixed tensors: hybrid mask, positions, num_items, and a fixed eval batch
    rng = np.random.default_rng(args.seed)
    fixed = []
    for s in samples:
        L = int(s["input_ids"].shape[0])
        mask2d = hybrid_block_causal_mask_dense(jnp.asarray(s["rbi"]), jnp.asarray(s["turn"]), L)
        mask4d = to_attn_mask4d(mask2d)
        pos = jnp.asarray(np.concatenate([np.arange(L), np.arange(L)])[None], dtype=jnp.int32)
        posB = jnp.broadcast_to(pos, (2, 2 * L))
        ni = noise_mod.num_items(s)
        resp = s["labels"] != -100
        fmask = resp & (~s["scaffold"]) & (np.arange(L) % 2 == 0)   # deterministic eval mask
        ev = noise_mod.make_batch(s, rng, fixed_mask=fmask)
        fixed.append(dict(mask4d=mask4d, posB=posB, num_items=ni, eval_batch=ev, L=L))

    print(f"loading model (source={args.source}, bf16) ...", flush=True)
    cfg = Qwen25TextConfig.fast_ddrive(dtype=jnp.bfloat16)
    snap = args.base_snap if args.source == "base" else SNAP
    model, info = load_fast_ddrive_text(snap, cfg=cfg, dtype=jnp.bfloat16, verbose=True)

    if not args.full_ft:
        n_wrap = apply_lora(model, LoRAConfig(rank=args.rank), rngs=nnx.Rngs(args.seed))
        n_frozen = freeze_non_lora_params(model)
        n_train = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
        print(f"LoRA: wrapped {n_wrap} linears, froze {n_frozen} base params; trainable={n_train/1e6:.2f}M")

    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, max(args.steps // 10, 1),
                                               max(args.steps - args.steps // 10, 1), args.lr * 0.1)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adafactor(learning_rate=sched, multiply_by_parameter_scale=True,
                                     min_dim_size_to_factor=128))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    def eval_all(tag):
        out = []
        for s, f in zip(samples, fixed):
            ifn, lfn, ol, w = f["eval_batch"]
            l = float(eval_loss(model, jnp.asarray(ifn), f["mask4d"], f["posB"],
                                jnp.asarray(lfn), jnp.asarray(w), jnp.asarray(ol), f["num_items"]))
            out.append(l)
        print(f"  [{tag}] fixed-eval loss per sample: " + ", ".join(f"{x:.4f}" for x in out))
        return out

    rows = [("step", "sample", "train_loss", "eval0", "eval1")]
    ev0 = eval_all("init")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        si = (step - 1) % len(samples)
        s, f = samples[si], fixed[si]
        ifn, lfn, ol, w = f["eval_batch"] if args.fixed_batch else noise_mod.make_batch(s, rng)
        l = float(train_step(model, optimizer, jnp.asarray(ifn), f["mask4d"], f["posB"],
                             jnp.asarray(lfn), jnp.asarray(w), jnp.asarray(ol), f["num_items"]))
        if step % 10 == 0 or step == 1:
            ev = eval_all(f"step {step}")
            rows.append((step, si, l, ev[0], ev[1] if len(ev) > 1 else ev[0]))
            print(f"step {step:3d} (sample {si}) train_loss {l:.4f}  [{(time.time()-t0)/step*1000:.0f} ms/step]", flush=True)
    evf = eval_all("final")

    with open(CSV, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"\nsaved {CSV}")
    print(f"INIT  eval: {[f'{x:.4f}' for x in ev0]}")
    print(f"FINAL eval: {[f'{x:.4f}' for x in evf]}")
    dec = all(evf[i] < ev0[i] for i in range(len(ev0)))
    print(f"\nPHASE3_{'PASS' if dec else 'FAIL'}: fixed-eval loss decreased on all samples "
          f"({[f'{ev0[i]:.3f}->{evf[i]:.3f}' for i in range(len(ev0))]})")
    sys.exit(0 if dec else 1)


if __name__ == "__main__":
    main()
