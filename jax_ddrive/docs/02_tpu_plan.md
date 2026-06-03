# Fast-dDrive JAX — TPU / MaxText scale-out plan (Phase 5)

The local 5090 validates correctness on a single device; this is the recipe to run the
same code on a Waymo TPU pod. The model is plain Flax NNX (`Qwen25TextModel` + the
Phase-4 ViT), so it slots into MaxText's mesh/sharding the same way `jax-mdlm-handoff`'s
`LLaDAModel` did (`code/models/llada.py:40-72` is the reference pattern).

## Mesh
- 2D mesh `('fsdp', 'tp')` via `jax.make_mesh((n_fsdp, n_tp), ('fsdp','tp'))`.
  - v5e-256 → e.g. `(64, 4)`; v6e → size to chips. Start pure-FSDP `(N, 1)` (simplest, scales the 3.75B fine).
- Wrap all jits in `with jax.sharding.use_mesh(mesh):` (nnx) — the reference learned to use
  `set_mesh`/`use_mesh`, NOT bare `PartitionSpec` without a mesh context (JaxRuntimeError otherwise).

## Param sharding (FSDP, matches llada.py ShardingConfig)
| Param | PartitionSpec (fsdp,tp) |
|---|---|
| `embed_tokens.embedding` [V,D] | `P(None, 'tp')` (vocab replicated, hidden TP) or `P('fsdp', None)` for pure-FSDP |
| `q/k/v/o_proj.kernel` [in,out] | `P('tp','fsdp')` / `P('fsdp','tp')` per llada pattern |
| `gate/up/down_proj.kernel` | `P('fsdp','tp')` / `P('tp','fsdp')` |
| norms, biases | `P('tp')` or replicated |
| activations [B,L,D] | `P('fsdp', None, 'tp')` via `with_sharding_constraint` |

Implementation options (increasing fidelity):
1. **FSDP-only, no model change** (fastest): shard `nnx.state(model, nnx.Param)` by putting each
   param's largest axis on `'fsdp'` via `jax.device_put(state, NamedSharding(mesh, pspec))`, jit the
   train step with matching `in/out_shardings`. The included `ddrive_jax/sharding.py` does this and is
   validated to be a no-op at mesh size 1 (so the 5090 path is unchanged).
2. **Explicit TP** (throughput): port `Linear`→`ShardedLinear` / `Embed`→`ShardedEmbedding`
   (copy `llada.py:122-158`'s `out_sharding=` plumbing) and annotate kernels with the table above.
3. **MaxText-native**: lift `diffusion/` + the SASD `loss_fn` into a MaxText fork exactly like
   `jax-mdlm-handoff/code-fork/README.md`'s 5-line `loss_fn` diff + the bidirectional attention-mode
   patch; wire Waymo data via grain. Checkpoints via MaxText's Orbax pipeline.

## Checkpointing
- Use `ddrive_jax/checkpoint.py` (Orbax `StandardCheckpointer`) — multi-host safe (the reference's
  pickle is single-host only). On a pod, point it at a GCS path; Orbax handles sharded save/restore.

## Memory at scale
- With FSDP across N chips, per-chip param/optimizer memory is `~1/N`, so **full fine-tune + AdamW**
  becomes feasible on a pod (no LoRA/remat needed — those were only for the single shared 5090).
  Keep `remat` available as a knob for very long sequences.

## Numerics
- Keep the loss/log-softmax in fp32 (`sasd_loss.py` already upcasts). bf16 params/activations are fine.
- TPU XLA is stricter than GPU: add NaN guards in eval; the Phase-1/2 parity gates (rel < 1e-3) should
  be re-run on TPU CPU/emulator before a big run.

## Validation ladder on TPU
1. mesh=1 on one chip: loss == the 5090 value for the same batch (sanity).
2. Single-step loss equal across `(N,1)` FSDP vs 1-chip (same seed/batch) within 1e-4.
3. Overfit the 2 samples → loss decreases (same as local Phase 3).
4. Real WOD-E2E training → ADE/RFS via the separate `autovla` TF metrics env.
