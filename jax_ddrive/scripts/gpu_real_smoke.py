"""Phase 4.5 — REAL Fast-dDrive model loss-decrease on the GPU, via the FSDP harness (single
device). Overfits ONE real batch to prove the *real* pretrained 3.75B text model + frozen ViT +
real Parquet data actually train through the harness and the loss decreases (grads flow,
optimizer steps, no NaN). This is the real-model counterpart to the proxy gate (B).

Run on the 5090:
  source jax_ddrive/scripts/jax_gpu_env.sh
  "$JAXPY" jax_ddrive/scripts/gpu_real_smoke.py
"""
import sys
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.train import train_tpu as T, dist
from ddrive_jax.data import grain_pipeline as gp

DATA = "/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd"   # 400-sample set (eager loader fits RAM)


def main():
    cfg = T.HarnessConfig(
        proxy=False, dtype=jnp.bfloat16, n_fsdp=1, n_tp=1,
        opt="adafactor", lr=1e-4, total_steps=40, warmup_steps=4,
        mrope_section=(16, 24, 24), seed=0,
    )
    # NOTE: pretrained model starts near-optimal (~0.98), so we overfit ONE batch at a clear lr
    # to prove the harness reduces loss on the REAL model. (Real multi-sample fine-tune = the pod.)
    dist.init_distributed()
    print("building REAL harness (loads pretrained 3.75B text + frozen ViT)...", flush=True)
    h = T.build_harness(cfg)
    loader = gp.make_sasd_loader(DATA, "train", per_host_batch=1, seed=0)
    batch = next(iter(loader))
    print(f"overfitting one real batch (sample {batch['sample_id']})...", flush=True)
    losses = []
    for st in range(1, 41):
        loss, _ = T.run_step(h, {**batch, "step": st})
        lv = float(loss)
        losses.append(lv)
        if st % 5 == 0 or st == 1:
            print(f"  step {st:3d} loss {lv:.4f}", flush=True)
    first = float(np.mean(losses[:3]))
    last = float(np.mean(losses[-3:]))
    ok = np.isfinite(last) and last < first - 0.05
    print(f"[done] loss {losses[0]:.4f} -> {losses[-1]:.4f} (first3 {first:.4f} -> last3 {last:.4f})")
    print("GPU_REAL_LOSSDECREASE_PASS" if ok else "GPU_REAL_LOSSDECREASE_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
