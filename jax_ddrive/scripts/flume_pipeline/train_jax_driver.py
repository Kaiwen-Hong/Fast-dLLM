"""Stage 2 (jax env, GPU/TPU): run the real-model SASD train step on the NEW dataloader's
batches and capture loss + logits for 1-step (both samples, bs=1) and a 10-step run.

Consumes the npz batches written by `materialize_batches.py` (the new msgpack dataloader's
output) and the validated `train_tpu` harness (real Fast-dDrive 3B text + frozen ViT + the
exact section/causal SASD loss). Because the batches are bit-exact to the prep-npz the
validated pipeline uses (gated in materialize), this is the same training step on the same
tensors — sourced through the new raw->AR->Grain path.

Outputs `results.json` (per-sample 1-step loss + logit fingerprints; 10-step loss curve;
post-10-step loss) and `logits_sig.npz` (top-5 ids/values over each sample's response span),
so GPU vs TPU (and vs the PyTorch oracle) can be compared exactly.

Run (jax env, `unset LD_LIBRARY_PATH`):
    export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive
    python train_jax_driver.py --batch_dir <materialized> --out_dir <results> \
        --opt adafactor --lr 2e-5 [--bf16]
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

sys.path.insert(0, os.environ.get("FASTDDRIVE_REPO",
                                  "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive"))
from ddrive_jax.train import train_tpu as _tt                                    # noqa: E402
from ddrive_jax.train.train_tpu import (build_harness, HarnessConfig, run_step,  # noqa: E402
                                        prepare_batch, IMAGE_TOK)
from ddrive_jax.train import dist                                                # noqa: E402
from ddrive_jax.diffusion.sasd_loss import section_weighted_ce, causal_ce        # noqa: E402

# train_tpu.SNAP is a hardcoded LOCAL path; on the TPU the release snapshot lives elsewhere.
# Honour FASTDDRIVE_SNAP (set by tpu_validate_flume.sh) so build_harness loads the real model
# from the staged dir. (jax_batch_inference already reads this env; train_tpu does not.)
_snap_env = os.environ.get("FASTDDRIVE_SNAP")
if _snap_env:
    _tt.SNAP = _snap_env

BATCH_ARRAYS = ["input_final", "labels_final", "original_labels", "weights",
                "position_ids", "rbi", "turn", "scaffold",
                "pixel_values", "image_grid_thw", "num_items"]


def load_batch(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    batch = {k: d[k] for k in BATCH_ARRAYS}
    batch["step"] = int(d["step"])
    batch["sample_id"] = [str(x) for x in d["sample_id"]]
    return batch


def forward_logits_loss(h, batch):
    """B=1 forward replicating train_tpu.per_sample_loss_sums but RETURNING logits.
    Returns (loss float, sec float, cau float, nl [2,L,V], cl [1,L,V], lfn0 [L])."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    pin = prepare_batch(batch, h.cfg, h.image_embeds_fn)            # host precompute, B=1
    # FSDP-typed params make matmul ambiguous outside the jitted step; replicate to P()
    # (free at n_fsdp=1) so this forward-only logit capture runs cleanly (mirrors the
    # `jax.reshard(params, P())` train_tpu does inside its shard_map).
    repl = NamedSharding(h.mesh, P())
    params_repl = jax.device_put(h.params, jax.tree.map(lambda _: repl, h.params))
    model = nnx.merge(h.graphdef, params_repl)                     # current params, replicated
    ifn = jnp.asarray(pin["input_final"])[0]                       # [2,2L]
    cos, sin = pin["cos"][0], pin["sin"][0]
    mask4d, img_pos, ie = pin["mask4d"][0], pin["img_pos"][0], pin["image_embeds"][0]
    Lh = ifn.shape[1] // 2

    def embed_row(r):
        e = model.embed_tokens(ifn[r])
        return e.at[img_pos].set(ie.astype(e.dtype))
    emb2 = jnp.stack([embed_row(0), embed_row(1)], 0)              # [2,2L,D]
    hidden = model.hidden_forward_mrope_cs(emb2, cos, sin, mask4d, remat=False)
    nl = model.attend(hidden[:, :Lh, :])                          # [2,L,V] noisy
    cl = model.attend(hidden[:1, Lh:, :])                         # [1,L,V] clean
    lfn = jnp.asarray(pin["labels_final"])[0]
    ol = jnp.asarray(pin["original_labels"])[0]
    w = jnp.asarray(pin["weights"])[0]
    sec = float(section_weighted_ce(nl, lfn, w, num_items=1.0))
    cau = float(causal_ce(cl, ol, num_items=1.0))
    num = float(np.asarray(pin["num_items"])[0])
    loss = (sec + cau) / max(num, 1.0)
    return loss, sec, cau, nl, cl, np.asarray(lfn[0])


def logit_fingerprint(nl, lfn0):
    """Compact, silicon-comparable fingerprint of the noisy-row logits over the response span."""
    nl0 = np.asarray(nl[0], np.float32)                           # [L, V]
    resp = np.where(lfn0 != -100)[0]
    top5_ids = np.argsort(-nl0[resp], axis=-1)[:, :5].astype(np.int32) if len(resp) else np.zeros((0, 5), np.int32)
    top5_val = np.take_along_axis(nl0[resp], top5_ids, axis=-1).astype(np.float32) if len(resp) else np.zeros((0, 5), np.float32)
    return {
        "logits_mean": float(nl0.mean()), "logits_std": float(nl0.std()),
        "logits_absmax": float(np.abs(nl0).max()), "logits_sum_f32": float(nl0.sum(dtype=np.float64)),
        "n_response": int(len(resp)),
        "top1_ids": top5_ids[:, 0].tolist() if len(resp) else [],
    }, top5_ids, top5_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_dir", required=True, help="dir of batch_*.npz from materialize_batches.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--opt", default="adafactor", choices=["adamw", "adafactor"])
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dist.init_distributed()
    dtype = jnp.bfloat16 if args.bf16 else jnp.float32
    meta = json.load(open(os.path.join(args.batch_dir, "meta.json")))
    batch_files = sorted(os.path.join(args.batch_dir, b["npz"]) for b in meta["batches"])
    batches = [load_batch(p) for p in batch_files]
    device = str(jax.devices()[0])
    print(f"[train-driver] {len(batches)} batches; device={device}; dtype={dtype.__name__}", flush=True)

    # warmup guard: optax.warmup_cosine_decay treats decay_steps as TOTAL length and uses
    # (decay_steps - warmup) internally, so tiny step counts can drive the cosine portion to 0
    # ("requires positive decay_steps"). For short validation runs use warmup=0 -> constant lr.
    warmup = max(args.steps // 10, 1) if args.steps >= 5 else 0
    cfg = HarnessConfig(proxy=False, dtype=dtype, n_fsdp=jax.device_count(), n_tp=1,
                        opt=args.opt, lr=args.lr, clip=1.0,
                        warmup_steps=warmup, total_steps=args.steps, seed=0)
    cfg.mrope_section = (16, 24, 24)
    h = build_harness(cfg)
    print("[train-driver] harness built (real 3B + frozen ViT)", flush=True)

    results = {"device": device, "dtype": dtype.__name__, "opt": args.opt, "lr": args.lr,
               "steps": args.steps, "one_step": [], "ten_step_losses": [], "post_train": []}
    sig_npz = {}

    # ---- Phase A: 1-step loss + logits per sample (bs=1) at INIT params -----------------
    # batch_00 = sample0, batch_01 = sample1 (round-robin from materialize, shuffle off)
    with dist.mesh_context(h.mesh):
        for i in range(min(2, len(batches))):
            b = batches[i]
            loss, sec, cau, nl, cl, lfn0 = forward_logits_loss(h, b)
            fp, t5i, t5v = logit_fingerprint(nl, lfn0)
            sid = b["sample_id"][0]
            results["one_step"].append({"sample_id": sid, "loss": loss, "sec_sum": sec,
                                        "causal_sum": cau, **fp})
            sig_npz[f"init_{i}_top5_ids"] = t5i
            sig_npz[f"init_{i}_top5_val"] = t5v
            print(f"[1-step] sample {i} ({sid}): loss={loss:.6f} sec={sec:.3f} cau={cau:.3f} "
                  f"logits_absmax={fp['logits_absmax']:.4f} top1[:8]={fp['top1_ids'][:8]}", flush=True)

    # ---- Phase B: 10 optimizer steps on the new-dataloader batches ----------------------
    for st in range(min(args.steps, len(batches))):
        loss, aux = run_step(h, batches[st])
        lv = float(loss)
        results["ten_step_losses"].append({"step": st, "sample_id": batches[st]["sample_id"][0], "loss": lv})
        print(f"[train] step {st}: sample={batches[st]['sample_id'][0]} loss={lv:.6f}", flush=True)

    # ---- post-train forward on both samples (final params) -----------------------------
    with dist.mesh_context(h.mesh):
        for i in range(min(2, len(batches))):
            b = batches[i]
            loss, sec, cau, nl, cl, lfn0 = forward_logits_loss(h, b)
            fp, t5i, t5v = logit_fingerprint(nl, lfn0)
            results["post_train"].append({"sample_id": b["sample_id"][0], "loss": loss, **fp})
            sig_npz[f"post_{i}_top5_ids"] = t5i
            sig_npz[f"post_{i}_top5_val"] = t5v
            print(f"[post-10] sample {i}: loss={loss:.6f} (was {results['one_step'][i]['loss']:.6f})", flush=True)

    json.dump(results, open(os.path.join(args.out_dir, "results.json"), "w"), indent=2)
    np.savez(os.path.join(args.out_dir, "logits_sig.npz"), **sig_npz)
    print(f"[train-driver] wrote results -> {args.out_dir}")
    print("TRAIN_JAX_DRIVER_DONE")


if __name__ == "__main__":
    main()
