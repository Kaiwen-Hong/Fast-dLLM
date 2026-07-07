# BLOCKERS / deferred items — overnight run 2026-07-07

1. **C5 train-infer consistency harness: DEFERRED (not run overnight).**
   The pinned-protocol comparison (design doc §B6) requires driving
   `sft_encode` / `sft_decode` / `GemmaARStateHandler.init_ar_state` outside
   their flax module context (explicit `.apply(..., method=...)` plumbing).
   Doing this faithfully is ~half-day work; a rushed version would compare the
   wrong tensors. Overnight GPU went to C3/C4/C6 instead. NOTE: the denoiser
   compute path is shared by construction (`SFTInferenceFn` and the training
   decode both route through `WrappedDiffusionGemmaNetwork.__call__`), so the
   residual C5 risk is concentrated in prefill/cache/position construction —
   exactly what the harness should pin. Sketch + scoping (<512 sliding window)
   in the design doc §B6 and ALIGNMENT_NOTES "open items".
2. **T2 batch sweep truncated:** b1+remat PASS (0.3s/step, peak 17.9GB);
   b2+remat was killed mid-compile (my kill, exit 143) and b4/b8 never ran.
   Production training probes b2/accum4 with automatic fallback to b1/accum8
   (run_e2b_overnight_chain.sh).
3. **Eval protocol reduction (per D5, documented):** overnight gates on
   steps=32 at the FINAL ckpt per variant; earlier ckpts (opt-50/100) + steps
   64/96 left for the morning if GPU time ran out.
4. **sliding-mask assert-equal-length unit test** (ALIGNMENT_NOTES row 10):
   not written; no mismatch can arise in the pre-expanded flow, test is
   belt-and-braces.
