# Small-scale end-to-end on the RTX 5090 — official Sudoku SFT harness

Proves the **clean** DiffusionGemma baseline runs the *full released training pipeline*
end-to-end on one RTX 5090: `kauldron.main` Trainer → orbax checkpoints → `eval_main`
AR-sampling eval → `SudokuAllMetrics`. Unlike `sft_smoke.py` (a hand-rolled loop over
synthetic data), this is the real harness on real data. Tiny random-init model, no
pretrained weights.

## What runs
- **Config:** `gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku_tiny.py`
  — the ONLY new file. A shrink of the shipped `sft_sudoku_full.py`: `embed_dim=128` /
  4 dense local-sliding layers / `enable_moe=False` / no `init_transform` (random init) /
  real Gemma4 vocab (262 144). Every framework file it imports
  (`data/sudoku/sudoku_data.py`, `hd/sft_model.py`, `eval_main.py`, `eval/sudoku_eval.py`,
  `safe_writer.py`, …) is the **pristine upstream framework — 0 edits**.
- **Data:** real Kaggle sudoku bagz on local disk
  (`/home/kaiwen/data/overnight_dgemma/data/sudoku_bagz/{sudoku_train,sudoku_eval}.bagz`).
- **Train:** `gpu_smoke/run_sudoku_train.sh` — 100 steps, ckpts @ 10/25/50/100.
- **Eval:**  `gpu_smoke/run_sudoku_eval.sh` — `eval_main --task=sudoku
  --eval_names=sample_ar_steps32` over those ckpts.

## Results (re-verified 2026-07-06, RTX 5090, env `dgemma-jax`)
- **Train** (`kauldron.main`): total **24.77 → 4.07** (diffusion 12.30→1.64,
  encoder 12.47→2.43), monotone, ~93 s wall, no NaN. Checkpoints `ckpt_{0,10,25,50,100}`
  + `train_complete.txt` written.
- **Eval** (`eval_main` AR sampler, exit 0 on all 4 ckpts): the full `SudokuAllMetrics`
  suite (`sudoku_accuracy`, `sudoku_cell_accuracy`, `sudoku_exact_mask_accuracy`,
  `sudoku_partial_mask_accuracy` × {overall, easy, medium, hard}) = **0.0** at every ckpt —
  EXPECTED at this scale. The AR text samples visibly shift from random-vocab gibberish
  (step 10) toward digit-grid tokens (step 25/50/100), i.e. the model learns to emit the
  answer *format*; actually *solving* sudoku needs the real 26B + released weights.

**Success criterion (met):** the official train+eval harness runs end-to-end on the 5090
and train losses decrease monotonically. Task accuracy is not a target for a tiny
random-init 100-step run.

Artifacts (NOT committed — live under `/home/kaiwen/data/dgemma_clean_e2e/`): checkpoints,
logs, tfevents.
