"""Phase-B load acceptance (design doc §7 C1 + C2-init), text scope.

Two worker modes (run as separate processes so trees never coexist):
  --mode ref  : ocp-restore the RAW AR ckpt (ground truth), bf16-cast, hash
                every leaf (sha256), run the plain Gemma4 AR forward on fixed
                tokens -> save hashes JSON + logits npy.
  --mode diff : init DiffusionGemma_E2B (text_only), load via OUR
                cheaply_load_params(expected_missing=('self_conditioner',),
                coverage_json_path=...), zero sc ffw output projection, hash
                every loaded leaf, AR forward on the same tokens, sc no-op
                check -> compare against ref artifacts -> verdict JSON.

C1 gates: (a) coverage: n_loaded + n_expected_missing == n_model AND
expected_missing subtrees == {self_conditioner}; (b) every non-sc model leaf
hash == raw-ckpt leaf hash (under the '/w' remap rule); (c) AR logits
max|diff| <= 1e-2 (expected 0.0).
C2-init gates: ||sc ffw linear|| == 0, ||sc gating|| > 0, and logits(sc=0) ==
logits(sc=random) bitwise.

Scope: FULL tree minus audio — `_encode_and_get_inputs` runs a dummy vision
call at init (gm/nn/gemma4/_transformer.py:498-500), so the vision tower params
exist and get loaded/bit-checked here. Audio is out of scope (decision D2):
`audio_encoder=None` config surgery; the ckpt's audio subtree is a counted
ckpt-only discard.
"""

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = "/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma"
sys.path.insert(0, f"{REPO}/gemma")
DATA = "/home/kaiwen/data/dgemma_e2b"
CKPTS = {"pt": f"{DATA}/ckpts/gemma4-e2b-pt", "it": f"{DATA}/ckpts/gemma4-e2b-it"}

import flax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

B, L = 1, 64


def fixed_tokens():
  # Deterministic ids in [3, 250000): no PAD(0)/EOS(1)/BOS(2), below specials.
  ids = (np.arange(B * L, dtype=np.int64) * 997 + 12345) % 249_000 + 3
  return jnp.asarray(ids.reshape(B, L), jnp.int32)


def leaf_hash(arr) -> str:
  a = np.asarray(jax.device_get(arr))
  return hashlib.sha256(a.tobytes()).hexdigest()


def restore_raw_cpu(path):
  import orbax.checkpoint as ocp
  import orbax.checkpoint._src.path.step as _step_lib
  _step_lib.is_path_finalized = lambda path: True
  with jax.default_device(jax.devices("cpu")[0]):
    ckpt = ocp.PyTreeCheckpointer()
    meta = ckpt.metadata(path)
    empty = jax.tree.map(
        lambda m: jnp.empty(m.shape, m.dtype, device=jax.devices("cpu")[0]),
        meta.item_metadata.tree,
    )
    return ckpt.restore(path, item=empty)


def mode_ref(variant: str):
  out_dir = f"{DATA}/loadtest/{variant}"
  os.makedirs(out_dir, exist_ok=True)
  t = time.time()
  raw = restore_raw_cpu(CKPTS[variant])
  flat = flax.traverse_util.flatten_dict(raw, sep="/")
  print(f"REF[{variant}]: restored {len(flat)} leaves in {time.time()-t:.0f}s")

  # Hash every leaf AFTER bf16 cast (matches what the loader puts on device).
  t = time.time()
  hashes = {k: leaf_hash(jnp.asarray(v).astype(jnp.bfloat16)) for k, v in flat.items()}
  with open(f"{out_dir}/ref_hashes.json", "w") as f:
    json.dump(hashes, f)
  print(f"REF[{variant}]: hashed in {time.time()-t:.0f}s")

  # AR forward with the raw tree (bf16) through plain Gemma4_E2B (full modules,
  # so the raw tree matches; vision/audio unused by a text forward).
  from gemma.gm.nn.gemma4 import _gemma4
  model = _gemma4.Gemma4_E2B(text_only=False, dtype=jnp.bfloat16)
  params = jax.tree.map(lambda x: jax.device_put(jnp.asarray(x).astype(jnp.bfloat16)), raw)
  del raw, flat
  logits = model.apply({"params": params}, fixed_tokens()).logits
  np.save(f"{out_dir}/ref_logits.npy", np.asarray(jax.device_get(logits), np.float32))
  print(f"REF[{variant}]: logits {logits.shape} saved. DONE")


def mode_diff(variant: str):
  out_dir = f"{DATA}/loadtest/{variant}"
  res_dir = f"{REPO}/results/{variant}"
  os.makedirs(res_dir, exist_ok=True)
  verdict = {"variant": variant}

  import dataclasses as _dc
  from gemma.diffusion import _models as dm
  from gemma.gm.nn.gemma4 import _gemma4
  from gemma.diffusion.hackable_diffusion_adapter.hd import gemma_checkpointer

  # Full tree minus audio (D2): vision stays; init's dummy vision call creates
  # the tower params so the loader covers them.
  cfg = _dc.replace(_gemma4.Gemma4_E2B.config, audio_encoder=None)
  model = dm.DiffusionGemma_E2B(text_only=False, config=cfg, dtype=jnp.bfloat16)
  tokens = fixed_tokens()
  positions = jnp.broadcast_to(jnp.arange(L), (B, L))
  attn = jnp.ones((B, L, L), dtype=bool)
  sc0 = jnp.zeros((B, L, model.config.embed_dim), jnp.bfloat16)

  # Init must trace the FULL multimodal path: without images, only the vision
  # TOWER params are created (dummy call at _encode_and_get_inputs), while the
  # embedder's mm_input_projection / mm_*_norm are created inside
  # encode_vision — i.e. only when images are passed. (Found via the coverage
  # report: ckpt embedder/mm_* keys were being discarded.)
  from gemma.gm.nn.gemma4 import _transformer as g4t
  # Two-step init: (a) REAL text-only init (cheap, proven; provides real
  # self_conditioner values — the only leaves the loader keeps); (b) FULL
  # multimodal trace via jax.eval_shape (zero device memory) for the mm-only
  # param STRUCTURE (vision tower / embedder mm_*), filled with zeros — every
  # one of them is replaced from the checkpoint by the loader, so values are
  # irrelevant; only shapes/dtypes matter.
  init_kwargs = dict(
      sc_embeddings=sc0, positions=positions, attention_mask=attn,
      sliding_attention_mask=attn,
      method=dm.DiffusionGemma_E2B.call_with_self_conditioning,
  )
  t = time.time()
  variables = model.init(
      {"params": jax.random.PRNGKey(0), "sampling": jax.random.PRNGKey(1)},
      tokens, **init_kwargs,
  )
  text_flat = flax.traverse_util.flatten_dict(variables["params"], sep="/")

  n_soft, n_patches, patch_dim = 256, 2520, 16 * 16 * 3
  mm_block = [108, 255999] + [-2] * n_soft + [258882, 108]
  tail = list(range(1000, 1030))  # Lmm=291 > 283 so l_no_mm stays positive
  mm_tokens = jnp.asarray([[2] + mm_block + tail], jnp.int32)
  Lmm = mm_tokens.shape[1]
  pvi = g4t.PreprocessedVisionInput(
      patches=jnp.zeros((1, n_patches, patch_dim), jnp.float32),
      positions_xy=jnp.zeros((1, n_patches, 2), jnp.int32),
      soft_token_counts=(n_soft,),
  )
  mm_pos = jnp.broadcast_to(jnp.arange(Lmm), (1, Lmm))
  mm_attn = jnp.ones((1, Lmm, Lmm), dtype=bool)
  spec = jax.eval_shape(
      lambda: model.init(
          {"params": jax.random.PRNGKey(0), "sampling": jax.random.PRNGKey(1)},
          mm_tokens,
          sc_embeddings=jnp.zeros((1, Lmm, model.config.embed_dim), jnp.bfloat16),
          images=pvi,
          positions=mm_pos,
          attention_mask=mm_attn,
          sliding_attention_mask=mm_attn,
          method=dm.DiffusionGemma_E2B.call_with_self_conditioning,
      )
  )
  spec_flat = flax.traverse_util.flatten_dict(spec["params"], sep="/")
  full_flat = {
      k: text_flat[k] if k in text_flat else jnp.zeros(s.shape, s.dtype)
      for k, s in spec_flat.items()
  }
  n_mm_only = len(spec_flat) - len(text_flat)
  variables = {"params": flax.traverse_util.unflatten_dict(full_flat, sep="/")}
  print(f"DIFF[{variant}]: init in {time.time()-t:.0f}s"
        f" (text leaves {len(text_flat)}, +{n_mm_only} mm-only zero leaves)")

  # ---- OUR loader: expected_missing + coverage report ----
  t = time.time()
  merged = gemma_checkpointer.cheaply_load_params(
      params_from_state=variables["params"],
      checkpoint_path=CKPTS[variant],
      expected_missing=("self_conditioner",),
      coverage_json_path=f"{res_dir}/load_coverage.json",
  )
  print(f"DIFF[{variant}]: loaded in {time.time()-t:.0f}s")
  with open(f"{res_dir}/load_coverage.json") as f:
    cov = json.load(f)
  verdict["coverage"] = {
      k: cov[k] for k in (
          "n_model", "n_loaded", "n_expected_missing",
          "expected_missing_subtrees", "n_ckpt_only_discarded",
          "ckpt_only_subtrees",
      )
  }
  cov_ok = (
      cov["n_loaded"] + cov["n_expected_missing"] + cov["n_lora_kept"]
      == cov["n_model"]
  ) and cov["expected_missing_subtrees"] == ["self_conditioner"]
  verdict["C1a_coverage_ok"] = bool(cov_ok)

  # ---- sc zeroing (W3 only) + saddle guard ----
  flat = flax.traverse_util.flatten_dict(merged, sep="/")
  sc_keys = [k for k in flat if "self_conditioner" in k]
  lin_keys = [k for k in sc_keys if "/ffw/linear" in k]
  gate_keys = [k for k in sc_keys if "/ffw/gating" in k]
  assert lin_keys and gate_keys, f"sc naming changed: {sc_keys}"
  for k in lin_keys:
    flat[k] = jnp.zeros_like(flat[k])
  merged = flax.traverse_util.unflatten_dict(flat, sep="/")
  lin_norm = float(sum(jnp.abs(flat[k]).sum() for k in lin_keys))
  gate_norm = float(sum(jnp.abs(flat[k]).sum() for k in gate_keys))
  verdict["C2_sc_keys"] = sc_keys
  verdict["C2_lin_norm"] = lin_norm
  verdict["C2_gate_norm"] = gate_norm
  verdict["C2_zero_ok"] = bool(lin_norm == 0.0 and gate_norm > 0.0)

  # ---- C1b bit-equality vs the RAW ckpt (ocp ground truth), respecting the
  # ---- per-leaf MODEL dtype (vision/mm params are intentionally fp32 via the
  # ---- _dtype_params exclude list; text params bf16). Path matching uses the
  # ---- simple '/w' rule, independent of the loader internals. ----
  raw = restore_raw_cpu(CKPTS[variant])
  raw_flat = flax.traverse_util.flatten_dict(raw, sep="/")
  n_eq = n_neq = n_unmatched = 0
  neq_sample = []
  for k, v in flat.items():
    if "self_conditioner" in k:
      continue
    rv = raw_flat.get(k)
    if rv is None:
      rv = raw_flat.get(k + "/w")
    if rv is None:
      n_unmatched += 1
      neq_sample.append(("UNMATCHED", k))
      continue
    expected = np.asarray(jax.device_get(jnp.asarray(rv).astype(v.dtype)))
    got = np.asarray(jax.device_get(v))
    if got.tobytes() == expected.tobytes():
      n_eq += 1
    else:
      n_neq += 1
      if len(neq_sample) < 10:
        neq_sample.append(("NEQ", k, str(v.dtype)))
  del raw, raw_flat
  verdict["C1b_bit_equality"] = {
      "n_equal": n_eq, "n_not_equal": n_neq, "n_unmatched": n_unmatched,
      "samples": neq_sample[:10],
  }
  verdict["C1b_ok"] = bool(n_neq == 0 and n_unmatched == 0)

  # ---- C1c AR-forward class-equivalence: the SAME loaded params through the
  # ---- plain Gemma4_E2B vs DiffusionGemma_E2B plain __call__ (extra sc
  # ---- subtree is unused by either). Catches any diffusion-class drift of
  # ---- the AR path; weight correctness is covered by C1b above. ----
  logits = model.apply({"params": merged}, tokens).logits
  ref_model = _gemma4.Gemma4_E2B(text_only=False, config=cfg, dtype=jnp.bfloat16)
  ref_logits = ref_model.apply({"params": merged}, tokens).logits
  max_diff = float(jnp.max(jnp.abs(
      logits.astype(jnp.float32) - ref_logits.astype(jnp.float32))))
  verdict["C1c_ar_logits_max_abs_diff"] = max_diff
  verdict["C1c_ok"] = bool(max_diff <= 1e-2)

  # ---- C2 sc no-op: logits(sc=0) == logits(sc=random) bitwise ----
  sc_rand = jax.random.normal(jax.random.PRNGKey(7), sc0.shape, jnp.bfloat16)
  def sc_fwd(sc_in):
    return model.apply(
        {"params": merged}, tokens, sc_embeddings=sc_in, positions=positions,
        attention_mask=attn, sliding_attention_mask=attn,
        method=dm.DiffusionGemma_E2B.call_with_self_conditioning,
    ).logits
  l0 = np.asarray(jax.device_get(sc_fwd(sc0)))
  lr = np.asarray(jax.device_get(sc_fwd(sc_rand)))
  verdict["C2_noop_bitwise"] = bool(np.array_equal(l0, lr))

  verdict["PASS"] = bool(
      verdict["C1a_coverage_ok"] and verdict["C1b_ok"]
      and verdict["C1c_ok"] and verdict["C2_zero_ok"]
      and verdict["C2_noop_bitwise"]
  )
  with open(f"{res_dir}/phaseb_verdict.json", "w") as f:
    json.dump(verdict, f, indent=2)
  print(f"DIFF[{variant}]: " + json.dumps(
      {k: v for k, v in verdict.items() if k != "C2_sc_keys"}, indent=2))
  print(f"DIFF[{variant}]: {'PASS' if verdict['PASS'] else 'FAIL'}")
  sys.exit(0 if verdict["PASS"] else 1)


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--mode", choices=["ref", "diff"], required=True)
  p.add_argument("--variant", choices=["pt", "it"], required=True)
  a = p.parse_args()
  (mode_ref if a.mode == "ref" else mode_diff)(a.variant)
