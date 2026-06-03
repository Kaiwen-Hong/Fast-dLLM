"""Phase-4b gate: multimodal forward parity. Feeds PyTorch's FUSED embeds + 3D M-RoPE
position_ids into the JAX text decoder (bidirectional mask) and compares hidden/logits.
Isolates M-RoPE + fusion-consumption from the ViT cuDNN seed. Gate: logits rel-max < 1e-3."""
import sys
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.models.qwen2_5_text import Qwen25TextConfig
from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_text

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
ORACLE = "/home/kaiwen/data/fast-ddrive/ref_logits/mm_oracle.npz"


def main():
    d = np.load(ORACLE)
    fused = jnp.asarray(d["fused_embeds"], jnp.float32)           # [1, L, D]
    pos3d = d["position_ids"][:, 0, :]                            # [3, L]
    L = fused.shape[1]
    print("fused", fused.shape, "pos3d", pos3d.shape)

    model, info = load_fast_ddrive_text(SNAP, Qwen25TextConfig.fast_ddrive(jnp.float32),
                                        dtype=jnp.float32, verbose=False)
    mask4d = jnp.ones((1, 1, L, L), bool)
    hid = model.hidden_forward_mrope(fused, pos3d, mask4d, mrope_section=(16, 24, 24))
    hid_np = np.asarray(hid, np.float32)
    logits = np.asarray(model.attend(hid), np.float32)

    def relmax(a, b): return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))
    hr = relmax(hid_np, d["hidden"]); lr = relmax(logits, d["logits"])
    top1 = float((logits.argmax(-1) == d["logits"].argmax(-1)).mean())
    print(f"\n=== PHASE 4b MULTIMODAL FORWARD PARITY (M-RoPE + fusion) ===")
    print(f"hidden rel-max {hr:.3e}  logits rel-max {lr:.3e}  top-1 {top1*100:.2f}%")
    gate = lr < 1e-3
    print(f"\nPHASE4b_MM_{'PASS' if gate else 'FAIL'} (logits rel-max {lr:.3e} {'<' if gate else '>='} 1e-3)")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
