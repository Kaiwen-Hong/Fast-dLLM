# eval_sasd GPU↔TPU parity — overnight findings (working notes)

Goal: confirm the JAX `eval_sasd` section-diffusion sampler produces the SAME output on TPU as on
GPU (so TPU inference is trustworthy); root-cause + fix any divergence. Model = NVIDIA release
Fast-dDrive (fp32 16 GB snapshot); inputs = 20 WOD-E2E val npz (`eval_inputs/`, production
precomputed-embeds path = ViT offline, sampler+text decoder only). Both precisions fp32 + bf16.

## Setup / method
- Harness `parity_eval.py`: loads release text decoder once, runs the production precomputed-embeds
  sampler per sample, dumps the FULL denoised token sequence (.npy) + trajectory + JSON.
- **Memory (5090, 32 GB RAM):** looping many samples in one process leaks XLA per-shape compiled-program
  device buffers → OOM after ~2 samples (varying L per sample). Fix = ONE sample per process (fresh
  process reclaims all VRAM). All 20 then succeed both precisions, no OOM.
- Comparator `parity_compare.py` / `parity_report.py`: token-level exact-match + agreement + first
  divergence index + trajectory max|Δ|.

## Results so far

### GPU reference (5090) — DONE
- fp32 20/20 ok, bf16 20/20 ok. Model produces REAL, varied trajectories (e.g. s04 [5.71,…],
  s05 [7.15,…], s08 [9.88,…]); s00/s03 happen to be near-0. (`valid_json` often False = JSON not
  fully closed, but the 5-waypoint trajectory parses — fine for parity.)
- **GPU fp32 is deterministic**: re-run of val_s00 fp32 = bit-identical tokens (agree 1.000000).
  → any backend difference found below is REAL, not run-to-run noise.

### Numeric sensitivity — fp32 vs bf16 (same GPU)
- mean token-agree **0.9847** (min 0.9454). The sampler IS sensitive to precision → the parity test
  is meaningful (not trivially passing).
- Mechanism (confirmed via first-divergence vs `rbi` block): the confidence-threshold unmasking
  (softmax conf > 0.9) flips on small logit diffs; an EARLY-block flip (rbi 0) cascades (s13/s17:
  ~62 divergent tokens, agree ~0.945); a LATE-block flip (rbi 4–5) barely propagates (2–15 tokens,
  agree 0.98–0.998); 2 samples identical (s09/s12). This is the same mechanism that would drive any
  GPU↔TPU divergence.

### Backend portability (fp32) — GPU vs CPU as a TPU proxy
TPU = another XLA backend; fp32 + `matmul_precision=highest` is true fp32 on all of CUDA / CPU / TPU.
CPU is a free local 3rd backend → a proxy for "is the sampler fp32-backend-portable?".
- **GPU-fp32 == CPU-fp32 EXACT (agree 1.000000) on ALL 20 val samples — 0 divergent tokens** —
  including the 3 HARDEST (s13, s17, s06 — the ones that cascade most under bf16, block-0
  first-divergence). Even where the confidence-cascade is most active under bf16, fp32 is bit-identical
  across XLA backends (CUDA↔CPU).
  → the sampler is **fp32-backend-portable**: fp32 backend diffs (~1e-6) are far below the 0.9
  confidence threshold, so they never flip an unmask decision; only bf16 rounding (~1e-2) does.

### Real TPU (v6e-1) — BLOCKED on capacity
- GCP free-trial v6e-1 unavailable: cross-zone (us-east5-a/b/c, us-east1-d, us-central1-a, us-south1-a)
  + backoff retry, multiple rounds, all WAITING_FOR_RESOURCES → no capacity. (Known external/transient
  issue, same as the docs' standing ≥8-chip open item.) Concluded **TPU_NO_CAPACITY** after 10 rounds
  / ~5 h / 6 zones; **never ACTIVE → zero cost, no VM ever created** (reaper armed only on ACTIVE).
- fp32 needs v6e (32 GB HBM); v5e-1 (16 GB) can't hold the 16 GB fp32 model. So only v6e retried.
- Full TPU harness is built + ready (`tpu_up.sh` retry, `tpu_run.sh`, `tpu_push.sh`, `tpu_down.sh`):
  the moment capacity frees it stages (model from GCS) + runs the identical per-sample eval + compares.

## Preliminary conclusion (to confirm with real TPU when capacity frees)
The eval_sasd sampler is **deterministic and fp32-backend-portable** (GPU==CPU bit-exact in fp32).
fp32 backend diffs (~1e-6) are far below the 0.9 confidence threshold that drives divergence, so
**TPU-fp32 inference is expected to match GPU-fp32** → trustworthy. bf16 is expected to diverge more
(the cascade is bf16-triggered) — for trustworthy TPU inference, prefer fp32 (or precomputed-fp32→bf16
embeds + fp32 sampler), consistent with the existing INFERENCE_DEPLOY embedding-parity policy.
