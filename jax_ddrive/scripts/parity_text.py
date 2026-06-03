"""Phase-1 parity gate: JAX text backbone vs PyTorch oracle.

Loads the oracle npz, builds the JAX Qwen25TextModel from the same checkpoint, runs
the same bidirectional text-only forward (mask=None => full attention, positions=arange),
and reports rel-diff on logits + hidden + top-1 agreement. Gate: logits rel-diff < 1e-3.
"""
import os
import sys

import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")  # no TF32: true fp32 matmuls
import jax.numpy as jnp

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.models.qwen2_5_text import Qwen25TextConfig
from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_text

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
ORACLE = "/home/kaiwen/data/fast-ddrive/ref_logits/text_oracle.npz"


def relmax(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))


def rel_l2(a, b):
    return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + 1e-9))


def main():
    d = np.load(ORACLE)
    ids = d["input_ids"]
    ref_logits, ref_hidden = d["logits"], d["hidden"]
    print("oracle:", ids.shape, "logits", ref_logits.shape)

    cfg = Qwen25TextConfig.fast_ddrive(dtype=jnp.float32)
    cfg.return_hidden_states = True
    model, info = load_fast_ddrive_text(SNAP, cfg=cfg, dtype=jnp.float32)
    assert not info["n_missing"], f"missing tensors: {info['n_missing'][:5]}"

    ids_j = jnp.asarray(ids)
    logits_j, hiddens = model(ids_j)               # mask=None => full bidirectional; positions=arange
    logits_j = np.asarray(logits_j, dtype=np.float32)
    hidden_j = np.asarray(hiddens[-1], dtype=np.float32)  # final post-norm hidden

    print("\n=== PARITY ===")
    print(f"hidden  rel-max {relmax(hidden_j, ref_hidden):.3e}  rel-L2 {rel_l2(hidden_j, ref_hidden):.3e}")
    lr = relmax(logits_j, ref_logits)
    print(f"logits  rel-max {lr:.3e}  rel-L2 {rel_l2(logits_j, ref_logits):.3e}")
    top1 = float((logits_j.argmax(-1) == ref_logits.argmax(-1)).mean())
    print(f"top-1 agreement: {top1*100:.2f}%")
    gate = lr < 1e-3
    print(f"\nPHASE1_PARITY_{'PASS' if gate else 'FAIL'} (logits rel-max {lr:.3e} {'<' if gate else '>='} 1e-3)")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
