# MM-SASD-step three-way parity (overnight)

Standalone harness that validates **dataset processing + model + the full SASD training step**
are numerically correct across the three implementations, on the **same inputs** (toy = the
2 real WOD-E2E driving samples in `fast_ddrive/data/example/sample.json`, 3 cameras → 720
image tokens, L=1856):

```
PyTorch oracle (HF release)   <->   ddrive_jax (NNX)   <->   maxtext-dlm-fork (production)
        numeric anchor                executable spec              TPU path
```

All files here are **NEW and standalone** — no existing repo code is edited. They import the
real pipeline modules read-only. Artifacts/logs live under `/home/kaiwen/data/parity_overnight/`.

## One command

```bash
bash jax_ddrive/scripts/mm_step_parity/run_overnight_parity.sh      # ~20-40 min
# sentinel: SASD_MM_STEP_PARITY_PASS
```
Knobs: `RES=200704` (toy image res → 720 tokens), `NSAMP=2`.

## Scripts (each prints its own PASS/FAIL sentinel; failures don't mask one another)

| script | env | what it validates |
|---|---|---|
| `capture_oracle_sasd_mm.py` | ddrive (GPU) | PyTorch oracle: freezes ALL inputs + builds the doubled mm-SASD step (shared pure-numpy `noise.make_batch` + deterministic mask) + forward + loss → saves bundle (inputs, fused embeds, full logits, loss). Also asserts `get_rope_index == prep` and mask invariants. |
| `verify_dataset.py` | jax (CPU) | dataset processing: prep npz == `decode_row(parquet)` == `decode_example(arrayrecord)` **bit-exact**, all 12 fields. |
| `parity_nnx.py` | jax (GPU) | NNX vs oracle: **Layer2** (frozen fused embeds→decoder, STRICT 1e-3), **Layer1** (NNX runs own ViT, 1e-2), **Layer1-CTRL** (NNX text + oracle img_emb, STRICT — proves fusion code + isolates ViT seed), **determinism** (make_batch bit-exact), per-section loss. |
| `parity_maxtext.py` | jax (CPU) | MaxText math/wiring: `mrope_cos_sin` & `hybrid_block_causal_mask_dense` vs NNX/oracle; the REAL `sasd_loss_from_logits` (vocab-tiled CE + flat=2b+r + global norm) vs PyTorch loss. |
| `parity_maxtext_fullfwd.py` | jax (CPU) | **MaxText REAL-3B FULL forward**: builds qwen2.5-3b Linen, loads real converted weights (434/434), runs the actual decoder on the doubled mm-SASD step (image scatter via `attention_metadata`) → logits/loss vs oracle. Closes the transitive-closure gap. |

## Design (informed by an adversarial Codex review)

- **Comparison boundary (B):** full SASD training-step forward, comparing doubled-sequence
  **logits AND** primary/complementary/total **loss**, decomposed into ordered sub-checks
  (determinism → mask → embeds → logits relmax/relL2/top1 → target-NLL → loss → per-section)
  so a failure localizes and cancellation can't mask a real divergence.
- **Same inputs guaranteed:** both stacks consume the SAME frozen bundle; the noiser is the
  SAME pure-numpy `noise.make_batch(fixed_mask=...)`, asserted bit-exact (`input_final` etc.).
- **Two freeze layers:** Layer2 (frozen PyTorch fused embeds) is the **strict** "decoder is a
  faithful port" gate; Layer1 (each stack runs its own ViT) is the **integration** gate with a
  looser logit tol because the ViT carries the cuDNN-Conv3d seed (see Findings).
- **fp32 everywhere** + `matmul_precision=highest` (MaxText uses a fp32 parity config, NOT the
  bf16 canonical). bf16 deployment drift is a separate concern.

## Results (both samples, all stacks — PASS)

| check | sample 0 | sample 1 |
|---|---|---|
| determinism (NNX make_batch == oracle, bit-exact) | ✅ | ✅ |
| get_rope_index (prep numpy == PyTorch model) | ✅ maxdiff 0 | ✅ maxdiff 0 |
| dataset round-trip parquet+AR vs npz (12 fields) | ✅ bit-exact | ✅ bit-exact |
| **NNX Layer2** (decoder) logits relmax / loss-total rel | 7.4e-5 / 4.0e-7 | 1.2e-4 / 9.9e-7 |
| **NNX Layer1-CTRL** (oracle img → NNX decoder) relmax | 7.4e-5 (== Layer2) | 1.2e-4 (== Layer2) |
| NNX Layer1 (own ViT) logits relmax / top1 / loss rel | 8.8e-3 / 100% / 7e-7 | 6.1e-3 / 99.8% / 1e-7 |
| ViT embeds cosine vs oracle | 1.000000 | 1.000000 |
| **MaxText math** mrope / mask / loss-total rel | 0.0 / 0 mism / 2.3e-7 | 0.0 / 0 mism / 3.9e-7 |
| **MaxText REAL-3B full fwd** logits relmax / top1 / loss rel | 1.0e-4 / 100% / 8e-7 | 1.7e-4 / 100% / 1.8e-6 |

Tolerances: logits relmax < 1e-3 (Layer2 / no-ViT), < 1e-2 (paths through ViT, matching the
existing `parity_vit.py` end-to-end ViT tol); loss rel < 1e-3; mask bit-equal; mrope rel < 1e-5;
ViT cosine ≥ 0.999.

### TPU validation (real v6e-1, `tpu_mm_validate.sh`)

Because XLA lowers differently on TPU vs GPU (project lesson: GPU-validated ≠ TPU-validated),
the data + NNX implementation were re-confirmed on a **real v6e-1** (us-east5-b):

| check | sample 0 | sample 1 |
|---|---|---|
| **data** parquet round-trip vs prep npz (12 fields) | ✅ bit-exact | ✅ bit-exact |
| **NNX determinism** (make_batch == oracle) | ✅ | ✅ |
| **NNX Layer2** (decoder) logits relmax / loss-total rel | 7.2e-5 / 7.5e-7 | 1.0e-4 / 1.7e-6 |
| **NNX Layer1-CTRL** (oracle img → decoder) relmax | 7.2e-5 (== Layer2) | 1.0e-4 (== Layer2) |
| NNX Layer1 (own ViT) relmax / top1 | 9.0e-3 / 100% | 6.6e-3 / 99.84% |
| ViT cosine | 1.000000 | 1.000000 |

The TPU numbers match the GPU numbers to within precision — the fp32 forward + SASD loss
reproduce the PyTorch oracle on TPU silicon, with the same (benign, ViT-seed) Layer1 behavior.
Provisioning uses the free-trial queued-resource pattern (patient ~20 min/zone wait — a QR is
free until ACTIVE — + auto-reaper). The TPU is torn down after the run (`tpu_down.sh`).
Staging deps: `jax[tpu] flax transformers ml_dtypes safetensors numpy pyarrow jaxtyping`
(jaxtyping is required by the NNX models and was the one staging gotcha).
MaxText arms are not run on TPU here (heavy full dep tree; already CPU-validated).

## Key debugging finding (mismatch root-caused — NOT a bug)

NNX **Layer1** logit relmax is ~6–9e-3 (> the 1e-3 used for the ViT-free path), while its loss
(~1e-7), top-1 (100%), per-section loss (<3e-5), and ViT-embed cosine (1.000000) are all
excellent. Root cause, proven by **Layer1-CTRL**: feeding the *oracle* image embeds through the
*NNX* text-embed + scatter + decoder path reproduces Layer2 **to the digit** (7.4e-5). Therefore:

1. the NNX `embed_tokens` + image scatter code is bit-faithful (CTRL == Layer2), and
2. the entire Layer1 excess is the **ViT cuDNN-Conv3d numerical seed** (`patch_embed` conv vs
   JAX matmul, ~2.6e-4 on embeds) amplified through 36 decoder layers — benign, the documented
   behavior behind `parity_vit.py`'s 1e-2 end-to-end ViT tolerance. It is **not** a bug in the
   PyTorch oracle, the NNX port, or this harness.

The MaxText full-forward uses the *oracle* image embeds (Layer2-style), so it stays at ~1e-4.

## Scope / caveats

- Toy = 2 real val-format samples (cover all 4 sections, nonzero weights, scaffold, im_end+\n
  label, 3-image 720-res, canonical L=1856). A fast smoke gate, not a coverage proof; parametrize
  `NSAMP` for a wider sweep.
- fp32 parity mode. The bf16 deployment-precision drift is intentionally out of scope here.
- MaxText full-forward feeds frozen oracle image embeds (isolates the decoder+weights). The
  trainable in-graph ViT path (`sasd_vit_trainable=true`) is covered separately by `parity_vit`
  + the in-graph ViT smoke; not re-run here.
