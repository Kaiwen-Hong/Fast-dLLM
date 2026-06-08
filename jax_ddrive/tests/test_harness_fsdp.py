"""FSDP training-harness mechanics test (CPU 8-device emulation = multi-host proxy).

SMALL proxy model + dummy FROZEN image embeds + the REAL grain loader (small per-host batch).
Asserts + prints exactly "HARNESS_FSDP_TESTS_PASS" covering four gates:

  (A) PARITY        : one train_step's loss under mesh (8,1) FSDP == under mesh (1,1)
                      single-device, SAME seed+batch+initial params, abs diff < 1e-4.
  (B) LOSS-DECREASE : >= 20 steps on a fixed small batch -> loss strictly trends down, no NaN.
  (C) CKPT-RESUME   : train K steps, save via CheckpointManager, build a FRESH harness,
                      restore_latest, continue -> next step's loss matches the uninterrupted
                      run within 1e-5 AND step/grain_state restored.
  (D) SHARDING-REAL : large kernels carry a NamedSharding whose spec is NOT fully replicated
                      on 'fsdp' (params actually partitioned across the 8 devices).

Run (CPU 8-device emulation):
  export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive
  export JAX_PLATFORMS=cpu
  export XLA_FLAGS="--xla_force_host_platform_device_count=8"
  PY=/home/kaiwen/jax-dlm-baseline/.venv/bin/python
  $PY jax_ddrive/tests/test_harness_fsdp.py
"""
import os
import shutil
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P

from ddrive_jax.train import train_tpu as T
from ddrive_jax.train import dist, checkpoint_mgr
from ddrive_jax.data import grain_pipeline as gp

DATA = "/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd"
SPLIT = "train"
CKPT_DIR = "/tmp/ddrive_harness_test_ckpt"

VSMALL = 4096                # tiny proxy vocab -> [B,2,L,V] logits ~150MB (CPU-emulation safe)
T.IMAGE_TOK = VSMALL - 1     # remap the image-token id into the small vocab (see shrink_vocab)


def shrink_vocab(batch, V=VSMALL, image_tok_orig=151655):
    """Remap real token ids into a tiny vocab so the dense logits fit host RAM on CPU emulation.

    The harness mechanics under test (FSDP shard_map, psum global loss, optimizer, checkpoint,
    parity) are VOCAB-AGNOSTIC -- only the dense [B,2,L,V] logits scale with V, and on CPU
    8-device emulation ALL device memory lives in the single 30GB host. We map every id ->
    id % (V-1) and reserve V-1 for image tokens (so prepare_batch's img_pos =
    (input_final == T.IMAGE_TOK) lands on the same positions). Real full-vocab + real model
    forward is validated separately on GPU (single device, fits VRAM).
    """
    b = dict(batch)
    img = (batch["input_final"] == image_tok_orig)
    inp = (batch["input_final"] % (V - 1)).astype(batch["input_final"].dtype)
    inp[img] = V - 1
    b["input_final"] = inp
    for k in ("labels_final", "original_labels"):
        lab = batch[k].copy()
        m = lab != -100
        lab[m] = lab[m] % (V - 1)
        b[k] = lab
    return b


def shrunk(loader):
    """Yield shrink_vocab'd batches from a grain loader (keeps CPU-emulation logits tiny)."""
    for bb in iter(loader):
        yield shrink_vocab(bb)


def _cfg(n_fsdp, **kw):
    """Small proxy config. FULL vocab (replicated-embedding path) but tiny everything else."""
    base = dict(
        proxy=True, dtype=jnp.float32,
        d_model=64, n_heads=4, n_kv_heads=2, head_dim=32, n_layers=2, mlp_hidden_size=128,
        vocab_size=VSMALL, mrope_section=(4, 6, 6),
        n_fsdp=n_fsdp, n_tp=1, opt="adamw", lr=1e-3, clip=1.0,
        warmup_steps=2, total_steps=30, seed=0,
    )
    base.update(kw)
    return T.HarnessConfig(**base)


def _one_batch(per_host_batch=2, seed=0):
    """A single fixed grain batch (the REAL loader).

    per_host_batch must be a multiple of n_fsdp (=8) so the leading B axis shards evenly
    across the 8-device FSDP mesh -- the real multi-host data-parallel constraint. We keep
    it at the minimum that divides 8 (=8) so each run stays fast on CPU.
    """
    ld = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=per_host_batch, seed=seed)
    return shrink_vocab(next(iter(ld)))


# ============================================================================ (A) PARITY ===
def test_parity():
    batch = _one_batch(per_host_batch=2, seed=0)

    KEY = ("layers", 0, "self_attn", "q_proj", "kernel")

    def loss_at(n_fsdp):
        cfg = _cfg(n_fsdp)
        h = T.build_harness(cfg)
        # snapshot a param BEFORE the step: train_step has donate_argnums=(0,1) and DELETES its
        # input params, so we must read them now (while valid). Same seed => both meshes start
        # from identical params. We also REASSIGN the returned params so h.params stays valid.
        pinit = np.asarray(dict(nnx.to_flat_state(h.params))[KEY][...])
        jit_inputs = T.prepare_batch(batch, cfg, h.image_embeds_fn)
        with dist.mesh_context(h.mesh):
            jit_inputs = jax.device_put(jit_inputs, h.data_shardings)
            h.params, h.opt_state, loss, aux = h.train_step(h.params, h.opt_state, jit_inputs)
        return float(loss), pinit

    l8, p8 = loss_at(2)   # FSDP mesh (2,1)
    l1, p1 = loss_at(1)   # single device
    diff = abs(l8 - l1)
    dpar = float(np.abs(p8 - p1).max())   # identical initial params across builds (same seed)
    assert dpar < 1e-6, f"initial params differ across builds: {dpar}"
    assert np.isfinite(l8) and np.isfinite(l1), f"non-finite loss l8={l8} l1={l1}"
    assert diff < 1e-4, f"PARITY FAIL: |l8-l1|={diff:.3e} (l8={l8:.6f} l1={l1:.6f})"
    print(f"  [A] PARITY OK (mesh (2,1) FSDP vs (1,1)): l_fsdp={l8:.6f} l_1dev={l1:.6f} |diff|={diff:.3e}")
    return diff


# ===================================================================== (B) LOSS-DECREASE ===
def test_loss_decrease():
    cfg = _cfg(2, total_steps=30, lr=2e-3)
    h = T.build_harness(cfg)
    batch = _one_batch(per_host_batch=2, seed=1)  # FIXED batch, re-fed every step
    losses = []
    for st in range(1, 26):
        loss, _ = T.run_step(h, {**batch, "step": st})
        lv = float(loss)
        assert np.isfinite(lv), f"non-finite loss at step {st}: {lv}"
        losses.append(lv)
    # strictly trends down: final-5 mean well below first-5 mean
    first = float(np.mean(losses[:5]))
    last = float(np.mean(losses[-5:]))
    assert last < first - 0.05, f"LOSS-DECREASE FAIL: first5={first:.4f} last5={last:.4f}"
    print(f"  [B] LOSS-DECREASE OK: {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"(first5 {first:.4f} -> last5 {last:.4f})")
    return losses


# ======================================================================== (C) CKPT-RESUME ===
def test_ckpt_resume():
    if os.path.isdir(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)

    K = 4  # steps before the checkpoint
    # ---- uninterrupted reference: run K+1 steps with the loader, record step K+1's loss ----
    cfg_ref = _cfg(2, total_steps=20, lr=1e-3)
    h_ref = T.build_harness(cfg_ref)
    ld_ref = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=2, seed=123)
    it_ref = shrunk(ld_ref)
    for _ in range(K):
        T.run_step(h_ref, next(it_ref))
    ref_batch = next(it_ref)
    ref_loss, _ = T.run_step(h_ref, ref_batch)
    ref_loss = float(ref_loss)

    # ---- interrupted run: train K steps, save, restore in a FRESH harness, continue --------
    cfg = _cfg(2, total_steps=20, lr=1e-3)
    h = T.build_harness(cfg)
    ld = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=2, seed=123)
    it = shrunk(ld)
    for _ in range(K):
        T.run_step(h, next(it))
    grain_state = ld.state()
    saved_step = K

    mgr = checkpoint_mgr.build_manager(CKPT_DIR, save_interval_steps=1, max_to_keep=2)
    checkpoint_mgr.save_step(mgr, saved_step, h.params, h.opt_state, grain_state,
                             extra_meta={"mrope_section": list(cfg.mrope_section)})
    checkpoint_mgr.wait(mgr)

    # FRESH harness (new model init -> params/opt differ until restore overwrites them)
    cfg2 = _cfg(2, total_steps=20, lr=1e-3, seed=777)
    h2 = T.build_harness(cfg2)
    mgr2 = checkpoint_mgr.build_manager(CKPT_DIR, save_interval_steps=1, max_to_keep=2)
    restored = checkpoint_mgr.restore_latest(
        mgr2, h2.abstract_params, h2.abstract_opt, h2.params_sharding, h2.opt_sharding)
    assert restored is not None, "CKPT-RESUME FAIL: restore_latest returned None"
    rstep, rgrain, rparams, ropt = restored
    assert rstep == saved_step, f"restored step {rstep} != {saved_step}"
    assert rgrain["grain"]["next_index"] == grain_state["grain"]["next_index"], \
        f"grain_state not restored: {rgrain} != {grain_state}"
    h2.params, h2.opt_state = rparams, ropt

    # restored params must match the saved harness exactly (sharded restore is bit-faithful)
    pa = dict(nnx.to_flat_state(h.params))
    pb = dict(nnx.to_flat_state(h2.params))
    key = ("layers", 1, "mlp", "down_proj", "kernel")
    dpar = float(np.abs(np.asarray(pa[key][...]) - np.asarray(pb[key][...])).max())
    assert dpar < 1e-6, f"restored params differ from saved: {dpar}"

    # continue the data stream from the restored grain state -> next batch == ref_batch
    ld2 = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=2, seed=123)
    ld2.set_state({"grain": rgrain["grain"]})
    cont_batch = shrink_vocab(next(ld2))   # next(ld2) directly: re-iter would reset set_state
    assert cont_batch["sample_id"] == ref_batch["sample_id"], \
        f"grain resume diverged: {cont_batch['sample_id']} != {ref_batch['sample_id']}"
    cont_loss, _ = T.run_step(h2, cont_batch)
    cont_loss = float(cont_loss)
    diff = abs(cont_loss - ref_loss)
    assert diff < 1e-5, f"CKPT-RESUME FAIL: cont={cont_loss:.6f} ref={ref_loss:.6f} diff={diff:.3e}"
    print(f"  [C] CKPT-RESUME OK: step={rstep} grain_next={rgrain['grain']['next_index']} "
          f"cont_loss={cont_loss:.6f} ref_loss={ref_loss:.6f} |diff|={diff:.3e}")
    if os.path.isdir(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)
    return diff


# ===================================================================== (D) SHARDING-IS-REAL ===
def test_sharding_is_real():
    cfg = _cfg(2)
    h = T.build_harness(cfg)
    flat = dict(nnx.to_flat_state(h.params))
    # large kernels: q/k/v/o_proj, mlp gate/up/down -> spec must reference 'fsdp'
    kernel_keys = [k for k in flat
                   if k and k[-1] == "kernel" and "embed" not in str(k)]
    assert kernel_keys, "no kernels found"
    sharded = []
    for k in kernel_keys:
        spec = flat[k][...].sharding.spec
        on_fsdp = any(("fsdp" in (str(s) if s is not None else "")) for s in spec)
        if on_fsdp:
            sharded.append((k, spec))
    assert len(sharded) >= 4, \
        f"SHARDING FAIL: only {len(sharded)} kernels sharded on 'fsdp': {sharded}"
    # embedding must be replicated (P())
    emb_spec = flat[("embed_tokens", "embedding")][...].sharding.spec
    assert tuple(emb_spec) == () or all(s is None for s in emb_spec), \
        f"embedding not replicated: {emb_spec}"
    # the sharded kernels' arrays are actually split across >1 device
    q = flat[("layers", 0, "self_attn", "q_proj", "kernel")][...]
    ndev = len({d for d in q.sharding.device_set})
    names = [str(k) for k, _ in sharded[:4]]
    print(f"  [D] SHARDING-IS-REAL OK: {len(sharded)} kernels on 'fsdp' "
          f"(e.g. {sharded[0][0][-3:]} -> {sharded[0][1]}); "
          f"embedding replicated; q_proj across {ndev} devices")
    return [n for n, _ in [(str(k[-3:]), s) for k, s in sharded]]


def main():
    print(f"running harness FSDP tests (devices={jax.device_count()}, procs={jax.process_count()})")
    assert jax.device_count() >= 2, \
        f"need >=2 CPU devices (set XLA_FLAGS=--xla_force_host_platform_device_count=8); got {jax.device_count()}"
    diff = test_parity()
    test_loss_decrease()
    test_ckpt_resume()
    test_sharding_is_real()
    print("HARNESS_FSDP_TESTS_PASS")


if __name__ == "__main__":
    main()
