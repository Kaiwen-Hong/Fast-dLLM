"""OVERNIGHT PARITY — Stage C: maxtext-dlm-fork SASD math/wiring vs the PyTorch oracle.

Standalone (does NOT edit existing code). Runs in the `jax` venv. Consumes the oracle bundle.

Codex finding #3/#7: the math arm is honestly labelled "math/input-wiring parity", NOT full
MaxText forward parity (a MaxText-only forward bug in embedding scatter / RoPE threading / qkv
layout / weight mapping would not be caught here — that needs the separate full-forward gate).

What this validates against the PyTorch oracle / NNX:
  mrope : maxtext.mrope_cos_sin(pos_doubled) == NNX.mrope_cos_sin  (bit-identical port)
  mask  : maxtext.hybrid_block_causal_mask_dense(rbi,turn,L) == oracle dense_mask  (bit-equal)
  loss  : the REAL production maxtext.sasd_loss_from_logits(<oracle logits>) reproduces the
          PyTorch primary/complementary/total loss (rel < 1e-3). This exercises the vocab-tiled
          online-logsumexp CE, the flat=2b+r reshape/split, section weighting, and global norm.

Both maxtext.sasd and ddrive_jax.qwen2_5_text.mrope are imported BY FILE PATH to avoid pulling
in the heavy package __init__ chains.

Usage:
  python parity_maxtext.py --bundle <dir/00000.bundle.npz>
"""
import argparse, importlib.util, json, os, sys
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp

ROOT = "/home/kaiwen/Desktop/research/Fast-dLLM"
MAXTEXT_SASD = f"{ROOT}/maxtext-dlm-fork/src/maxtext/diffusion/sasd.py"
sys.path.insert(0, f"{ROOT}/jax_ddrive")
from ddrive_jax.models.qwen2_5_text import mrope_cos_sin as nnx_mrope_cos_sin  # NNX port (pkg import)
TOL_LOSS, TOL_MROPE = 1e-3, 1e-5


def _import_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    stem = args.bundle.replace(".bundle.npz", "")
    d = np.load(args.bundle)
    L = int(d["L"]); num_items = float(d["num_items"])
    out = {"bundle": args.bundle, "L": L, "checks": {}}
    print(f"[C] {os.path.basename(stem)}: L={L} num_items={num_items} torch total={float(d['total']):.6f}", flush=True)

    mt = _import_by_path("mt_sasd", MAXTEXT_SASD)

    # ---- 1) M-RoPE: MaxText vs NNX (both ports of the same contiguous-chunk M-RoPE) ----
    pos_doubled = d["pos_doubled"]                                    # [3, 2L]
    mt_cos, mt_sin = mt.mrope_cos_sin(pos_doubled, 128, 1e6, (16, 24, 24))
    nx_cos, nx_sin = nnx_mrope_cos_sin(pos_doubled, 128, 1e6, (16, 24, 24))
    mt_cos, mt_sin = np.asarray(mt_cos), np.asarray(mt_sin)
    nx_cos, nx_sin = np.asarray(nx_cos), np.asarray(nx_sin)
    mrope_cos_rel = float(np.abs(mt_cos - nx_cos).max() / (np.abs(nx_cos).max() + 1e-9))
    mrope_sin_rel = float(np.abs(mt_sin - nx_sin).max() / (np.abs(nx_sin).max() + 1e-9))
    mrope_ok = mrope_cos_rel < TOL_MROPE and mrope_sin_rel < TOL_MROPE
    out["checks"]["mrope"] = {"pass": mrope_ok, "cos_rel": mrope_cos_rel, "sin_rel": mrope_sin_rel}
    print(f"[C] mrope MaxText vs NNX: cos_rel {mrope_cos_rel:.2e} sin_rel {mrope_sin_rel:.2e} -> {mrope_ok}", flush=True)

    # ---- 2) hybrid mask: MaxText vs oracle dense mask (bit-equal) ----
    mt_mask = np.asarray(mt.hybrid_block_causal_mask_dense(
        jnp.asarray(d["rbi"]), jnp.asarray(d["turn"]), L)).astype(bool)
    ref_mask = d["dense_mask"].astype(bool)
    mask_ok = bool((mt_mask == ref_mask).all())
    mism = int((mt_mask != ref_mask).sum())
    out["checks"]["mask"] = {"pass": mask_ok, "mismatches": mism}
    print(f"[C] hybrid mask MaxText == oracle: {mask_ok} (mismatches={mism})", flush=True)

    # ---- 3) loss: the REAL sasd_loss_from_logits on the captured oracle logits ----
    # Assemble the full doubled logits [2B=2, 2L, V]: row0=[noisy0|clean0], row1=[noisy1|dummy].
    # sasd_loss_from_logits uses only lg[:,:,:L] (both rows) + lg[:,0,L:] (row0) -> dummy is unused.
    noisy = np.load(f"{stem}.noisy_logits.npy", mmap_mode="r")        # [2, L, V]
    clean = np.load(f"{stem}.clean_logits.npy", mmap_mode="r")        # [1, L, V]
    V = noisy.shape[-1]
    noisy_j = jnp.asarray(np.asarray(noisy))                          # [2,L,V]
    clean0 = jnp.asarray(np.asarray(clean[0]))                        # [L,V]
    dummy = jnp.zeros((L, V), jnp.float32)
    row0 = jnp.concatenate([noisy_j[0], clean0], axis=0)             # [2L,V]
    row1 = jnp.concatenate([noisy_j[1], dummy], axis=0)             # [2L,V]
    full = jnp.stack([row0, row1], axis=0)                           # [2,2L,V] == [2B,2L,V]
    labels_final = jnp.asarray(d["labels_final"])[None]              # [1,2,L]
    original_labels = jnp.asarray(d["original_labels"])[None]        # [1,1,L]
    weights = jnp.asarray(d["weights"])[None]                       # [1,2,L]
    ni = jnp.asarray([num_items], jnp.float32)                       # [1]
    loss, sec_sum, cau_sum, tw = mt.sasd_loss_from_logits(
        full, labels_final, original_labels, weights, ni, B=1, L=L)
    mt_total = float(loss); mt_primary = float(sec_sum) / num_items; mt_comp = float(cau_sum) / num_items
    to, po, co = float(d["total"]), float(d["primary"]), float(d["complementary"])
    rel = lambda a, b: abs(a - b) / (abs(b) + 1e-9)
    loss_res = {"primary_maxtext": mt_primary, "primary_torch": po, "primary_rel": rel(mt_primary, po),
                "complementary_maxtext": mt_comp, "complementary_torch": co, "complementary_rel": rel(mt_comp, co),
                "total_maxtext": mt_total, "total_torch": to, "total_rel": rel(mt_total, to)}
    loss_ok = loss_res["total_rel"] < TOL_LOSS and loss_res["primary_rel"] < TOL_LOSS and loss_res["complementary_rel"] < TOL_LOSS
    loss_res["pass"] = bool(loss_ok)
    out["checks"]["loss"] = loss_res
    print(f"[C] sasd_loss_from_logits: primary {mt_primary:.6f}/{po:.6f} rel {rel(mt_primary,po):.2e} | "
          f"comp {mt_comp:.6f}/{co:.6f} rel {rel(mt_comp,co):.2e} | total {mt_total:.6f}/{to:.6f} rel {rel(mt_total,to):.2e}",
          flush=True)

    gate = mrope_ok and mask_ok and loss_ok
    out["pass"] = bool(gate)
    rp = args.report or f"{stem}.parity_maxtext.json"
    json.dump(out, open(rp, "w"), indent=2)
    print(f"\n[C] report -> {rp}")
    print(f"SASD_MM_MAXTEXT_MATH_PARITY_{'PASS' if gate else 'FAIL'} "
          f"(mrope={mrope_ok} mask={mask_ok} loss={loss_ok})")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
