"""Phase-2 gate: JAX SASD loss vs the PyTorch oracle (capture_oracle_sasd.py).

(1) mask unit test: my masks.hybrid_block_causal_mask_dense(rbi,turn,L) == torch dense mask.
(2) loss gate: feed the captured doubled inputs/mask/positions/weights to the JAX backbone,
    compute primary+complementary via sasd_loss, compare to torch. Gate: total rel-diff < 1e-3.
"""
import sys
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.models.qwen2_5_text import Qwen25TextConfig
from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_text
from ddrive_jax.diffusion.masks import hybrid_block_causal_mask_dense, to_attn_mask4d
from ddrive_jax.diffusion.sasd_loss import section_weighted_ce, causal_ce

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
ORACLE = "/home/kaiwen/data/fast-ddrive/ref_logits/sasd_oracle.npz"


def main():
    d = np.load(ORACLE)
    L = int(d["L"]); num_items = float(d["num_items"])
    rbi, turn = jnp.asarray(d["rbi"]), jnp.asarray(d["turn"])
    print(f"L={L} num_items={num_items} torch total={float(d['total']):.6f}")

    # (1) mask unit test
    my_mask = np.asarray(hybrid_block_causal_mask_dense(rbi, turn, L))
    ref_mask = d["dense_mask"].astype(bool)
    mask_ok = bool((my_mask == ref_mask).all())
    print(f"[mask] my hybrid mask == torch dense mask: {mask_ok} "
          f"(mismatches={int((my_mask != ref_mask).sum())})")

    # (2) loss gate
    cfg = Qwen25TextConfig.fast_ddrive(dtype=jnp.float32)
    model, info = load_fast_ddrive_text(SNAP, cfg=cfg, dtype=jnp.float32, verbose=False)
    input_final = jnp.asarray(d["input_final"])          # [2, 2L]
    labels_final = jnp.asarray(d["labels_final"])        # [2, L]
    original_labels = jnp.asarray(d["original_labels"])  # [1, L]
    weights = jnp.asarray(d["weights"])                  # [2, L]
    pos = jnp.asarray(d["position_ids"])                 # [1, 2L]
    mask4d = to_attn_mask4d(jnp.asarray(ref_mask))       # [1,1,2L,2L]
    posB = jnp.broadcast_to(pos, (input_final.shape[0], 2 * L))

    hidden = model.hidden_forward(input_final, mask4d, posB)   # [2, 2L, D] (no full lm_head)
    noisy_logits = model.attend(hidden[:, :L, :])              # [2, L, V]
    clean_logits = model.attend(hidden[:1, L:, :])             # [1, L, V]
    primary = float(section_weighted_ce(noisy_logits, labels_final, weights, num_items=num_items))
    comp = float(causal_ce(clean_logits, original_labels, num_items=num_items))
    total = primary + comp

    tp, tc, tt = float(d["primary"]), float(d["complementary"]), float(d["total"])
    rel = abs(total - tt) / (abs(tt) + 1e-9)
    print("\n=== PHASE 2 LOSS PARITY ===")
    print(f"primary       jax {primary:.6f}  torch {tp:.6f}  rel {abs(primary-tp)/(abs(tp)+1e-9):.2e}")
    print(f"complementary jax {comp:.6f}  torch {tc:.6f}  rel {abs(comp-tc)/(abs(tc)+1e-9):.2e}")
    print(f"total         jax {total:.6f}  torch {tt:.6f}  rel {rel:.2e}")
    gate = mask_ok and rel < 1e-3
    print(f"\nPHASE2_PARITY_{'PASS' if gate else 'FAIL'} (mask_ok={mask_ok}, total rel {rel:.2e} {'<' if rel<1e-3 else '>='} 1e-3)")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
