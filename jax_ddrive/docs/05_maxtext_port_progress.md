# Phase 7 — Fast-dDrive SASD on MaxText (native) — overnight build log

**Started 2026-06-06.** Goal: train Fast-dDrive (Qwen2.5-VL-3B block-diffusion VLA) **natively in
MaxText** so it deploys on Waymo TPU infra. Living log; updated every turn. Honesty rule: label
verified vs in-progress vs not-done.

## Locked decisions (user, 2026-06-06)
1. **Model**: native MaxText Qwen2.5 decoder (graft SASD onto it), NOT reuse-our-NNX.
2. **Vision**: full VLA — wire frozen-ViT image-embed injection.
3. **Weights**: load pretrained Fast-dDrive (A2D-style HF→MaxText mapping + MASK mean-init).
4. **TPU**: spending authorized, **hard cap = the $300 trial credit; I aim <$20** and never leave a VM.

## 🔴 MONEY-SAFETY RULES (non-negotiable)
- Hard cap $300; **target <$20**. Cheapest shape that proves the point first.
- **Every** TPU script: `trap cleanup EXIT INT TERM` auto-delete + standalone teardown/check scripts.
- **$0.04 `v5litepod-1` probe** before any real slice (confirms a trial acct can launch; no charge if it fails).
- GCP **budget alert** set (tripwire emails). After EVERY TPU touch: `tpu_check` → both lists empty.
- A forgotten `v5litepod-16` ≈ **$460/day** — the #1 risk. Never end a turn with a live slice unaccounted for.
- Account: `robosuite1998@gmail.com`, project `project-8a53f5ab-2ea2-4892-a78`, zone `us-east5-a`.

## Foundations (verified before Phase 7)
- MaxText fork `/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork` (v0.2.1) imports in the jax venv via `PYTHONPATH=src`.
- Already has: `diffusion/mdlm.py` (Sahoo MDLM loss), `objective:"mdlm"` branch in `trainers/pre_train/train.py`, bidirectional attn via `AttentionType.FULL`, grain+HF+parquet data, Qwen2.5 {1.5b,7b,14b} configs, BD3LM block-causal mask in sibling `maxtext-dlm`.
- Reference recipe: `DLM-policy4AV/jax-mdlm-handoff` (LLaDA-in-MaxText: ~10-line loss_fn diff, attn patch, a2d weight load).
- Our validated NNX SASD source of truth: `ddrive_jax/diffusion/{sasd_loss,noise,masks}.py`, model `models/qwen2_5_text.py` + `vision_qwen25vl.py`, data `data/{parquet_dataset,grain_pipeline}.py`. Dataset: `kaiwen2/wod-e2e-fast-ddrive-sasd-50k` (private HF) + local `/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd{,_50k}`.

## Phase 7 breakdown
| # | Task | Status |
|---|---|---|
| 7.0 | MaxText synthetic train smoke (CPU) + qwen2.5-3b config | ✅ DONE (+ fixed a real fork bug) |
| 7.1a | Port SASD loss/noise/hybrid-block-mask → `maxtext/diffusion/sasd.py`; **bit-exact parity vs NNX** | ✅ DONE & VERIFIED (286/286 byte-exact) |
| 7.1b | Wire `objective:"sasd"` into `trainers/pre_train/train.py` (doubled seq, 3D M-RoPE, hybrid mask, section loss) | ✅ DONE & VERIFIED |
| 7.2 | Waymo Parquet → MaxText grain input (SASD fields) | ⏳ |
| 7.3 | Load Fast-dDrive Qwen2.5 text weights → MaxText qwen2.5-3b; **forward parity vs NNX** | ✅ DONE & VERIFIED (100% top-1, mean 1.9e-3, 434/434 leaves) |

### Key integration fact (7.1b)
MaxText's native M-RoPE (Qwen3-Omni Thinker) uses **interleaved** `[T,H,T,W,…]` section-mixing; Fast-dDrive uses
**contiguous chunks** `[TTTT|HHHHHH|WWWWWW]` — NOT bit-equal (diff ~1.67). Solution implemented: **bypass
MaxText RoPE, inject precomputed cos/sin** (`diffusion/sasd.mrope_cos_sin`, bit-exact to NNX) + inject the
hybrid 4D mask (skip `generate_attention_mask`, `jnp.where(mask, logits, min)`). The `objective:"sasd"` branch
in `train.py` flattens `[B,2,2L]→[2B,2L]`, runs the native qwen2.5 decoder, computes section+causal loss.
**Loss bit-exact vs NNX** (section/causal/total). Files: train.py, layers/{attentions,attention_op}.py,
configs/{types.py,base.yml}, diffusion/sasd.py(+mrope_cos_sin,apply_rope_full,prepare_sasd_inputs,sasd_loss_from_logits).
| 7.4 | Frozen-ViT image-embed injection + **full-VLA forward/loss parity vs NNX harness** | ✅ DONE & VERIFIED (rel 1.7e-4, 336 img-tok) |
| 7.5 | End-to-end MaxText SASD training: **loss-decrease** on real Waymo (GPU, adafactor/bf16/remat) | ✅ DONE & VERIFIED (0.98→0.64, 40 steps, 18.5GB VRAM) |
| 7.6 | TPU validation (frugal, safe; cross-host mechanics) | ⏳ |

## Verification ladder (honest)
- 7.1 SASD loss parity vs NNX `sasd_loss.py` (bit-exact on a real batch) ← strongest math gate
- 7.3 forward-logits parity vs NNX `Qwen25TextModel` (within bf16 tol)
- 7.5 loss-decrease on real data + matches NNX harness loss trajectory
- 7.6 TPU: `devices==N, procs==hosts`, 10 steps finite, multi-host Orbax ckpt

## Engine / watchers
- Chained background workflows (each re-invokes me) + RAM watchdog on heavy jobs (30 GB, no swap).
- TPU-liveness watcher after any TPU launch; budget alert as tripwire.

## Turn log
- T1 (2026-06-06): Recon — MaxText-DLM fork found + mapped (already has MDLM+bidir-attn+grain+Qwen2.5); jax-mdlm-handoff recipe read. User answers locked (native/full-VLA/pretrained/TPU-authorized). Wrote this doc; set budget alert; launched 7.0+7.1.
- T2 (2026-06-06): W1 returned: **7.0 ✅** (MaxText synthetic smoke trains on CPU; fixed a real pre-existing fork bug — `objective`/`mask_token_id`/`pad_token_id` were in base.yml but unregistered in the Pydantic schema → fork couldn't start ANY run; fixed in `configs/types.py` + added qwen2.5-3b config). **7.1a ✅ verified** (`maxtext/diffusion/sasd.py` bit-exact port; adversarial 286/286 byte-exact). **Permission fix**: workflow blocked ~9h on a subagent approval prompt (subagents didn't inherit the CLI --dangerously-skip-permissions); user chose persist → set `defaultMode:bypassPermissions` in `~/.claude/settings.json`; **probe confirms subagents now bypass (BYPASS_OK, 10s)**. Launching 7.1b (wire objective:sasd into train.py).
- T3 (2026-06-06): W2 returned: **7.1b ✅ verified** — objective:sasd wired into MaxText native qwen2.5 decoder; M-RoPE incompatibility found + solved (inject precomputed cos/sin); **loss bit-exact vs NNX**; re-run by me (SASD_TRAIN_STEP_PASS). Launched W3 (weight load + forward parity, GPU).
- T4 (2026-06-06): W3 returned: **7.3 ✅ verified** — pretrained Fast-dDrive weights load into MaxText qwen2.5-3b (434/434 leaves, 3.086B); forward **100% top-1 vs NNX** (mean 1.9e-3), re-run by me on GPU (SASD_WEIGHT_PARITY_PASS). Agent fixed 2 real fp32 fidelity bugs (bf16 logit-cast, TF32 ruled out). Launched W4 (ViT injection + full-VLA loss parity, GPU). No money spent; no TPU running.
- T5 (2026-06-06): W4 returned: **7.4 ✅ verified** — full-VLA SASD loss (real weights + frozen ViT + doubled seq + M-RoPE + hybrid mask) matches NNX harness; build rel 2.2e-4, independent verifier (diff sample, hand-coded NNX) rel 1.3e-5; **re-run by me on GPU: rel 1.7e-4 (SASD_VLA_PARITY_PASS)**. 336 img-tokens/sample, positions bit-equal. **The MaxText port is now faithful end-to-end (forward+loss).** ViT injection in `decoders.py _apply_embedding` (scatter at image-token positions); inert when no image embeds (regression-safe). Hit a real 30GB-host OOM → forced subprocess-phased loading. Launched W5 (e2e loss-decrease, GPU). No money spent; no TPU.
- T6 (2026-06-06): W5 (loss-decrease) initially **STUCK** — user flagged GPU idle; I found it crashed on 2 jit-training bugs the eager parity tests missed: (1) `train.py:122 int(data["sasd_B"][0])` (tracer→int) → fixed to derive B/L from static shapes; (2) test stripped `params["params"]` → `model.apply` KeyError 'params' on tied output head → fixed to keep `{"params":...}` wrapper. Re-ran from MAIN loop (clean bg + watchdog, no subagent poll-loop): **7.5 ✅ loss 0.982778→0.636391 over 40 steps, no NaN, peak VRAM 18.5GB (SASD_TRAIN_LOSSDECREASE_PASS)** — matches NNX smoke (0.985→0.598). Lesson: run decisive GPU steps from main loop, not subagent self-poll-loops. Next: 7.2 (deployable maxtext.train entry) → 7.6 (frugal TPU).
