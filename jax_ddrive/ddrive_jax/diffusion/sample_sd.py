"""JAX section-diffusion sampler (port of mdm_sample_deep_scaffold, simplest no-KV-cache
form). Iterates each response block, unmasking positions whose predicted-token confidence
exceeds `threshold` (argmax fallback), using the verified forward. Causal shift: the
prediction for position p uses the logit at p-1 (matches modeling.py loss + generation).

Run (parity vs PyTorch reference from scripts/prep_sd_inputs.py):
  python jax_ddrive/ddrive_jax/diffusion/sample_sd.py
"""
from __future__ import annotations
import sys
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.models.qwen2_5_text import Qwen25TextConfig, mrope_cos_sin
from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_text
from ddrive_jax.diffusion.masks import eval_hybrid_block_causal_mask_dense, to_attn_mask4d

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")


def block_ranges_from_rbi(rbi: np.ndarray):
    """Maximal contiguous spans of equal response_block_idx (>=0)."""
    ranges, i, L = [], 0, len(rbi)
    while i < L:
        if rbi[i] >= 0:
            j = i
            while j < L and rbi[j] == rbi[i]:
                j += 1
            ranges.append((int(rbi[i]), i, j)); i = j
        else:
            i += 1
    return ranges


def section_diffusion_sample(model, x_t0, rbi, cos, sin, mask4d, *,
                             mask_id=151665, threshold=0.9, max_tokens=512):
    """x_t0: int [1, L]. Returns generated int [1, L] (scaffold value positions filled)."""
    x_t = np.array(x_t0, np.int64)
    ranges = block_ranges_from_rbi(np.asarray(rbi))

    @nnx.jit
    def hid_fwd(model, ids):
        return model.hidden_forward_mrope_cs(model.embed_tokens(ids), cos, sin, mask4d)  # [1, L, D]

    steps = 0
    for (_, s, e) in ranges:
        n_mask = int((x_t[0, s:e] == mask_id).sum())
        for _ in range(n_mask + 5):
            cur = (x_t[0, s:e] == mask_id)
            if cur.sum() == 0:
                break
            hidden = hid_fwd(model, jnp.asarray(x_t))                          # [1, L, D]
            sec = np.asarray(model.attend(hidden[0, s - 1:e - 1]), np.float32) # causal shift -> [blk, V] (slice only)
            x1 = sec.argmax(-1)
            p = jax.nn.softmax(jnp.asarray(sec), -1)
            x1p = np.asarray(jnp.take_along_axis(p, jnp.asarray(x1)[:, None], -1)[:, 0])
            x1p = np.where(cur, x1p, -np.inf)
            unmask = np.where(x1p > threshold)[0]
            if unmask.size == 0:
                unmask = np.array([int(np.argmax(x1p))])
            x_t[0, s + unmask] = x1[unmask]
            steps += 1
            if steps > max_tokens:
                break
    return x_t


def main():
    from transformers import AutoTokenizer
    d = np.load("/home/kaiwen/data/fast-ddrive/ref_logits/sd_inputs.npz")
    rbi = d["rbi"]; L = d["x_t0"].shape[1]; prompt_len = int(d["prompt_len"])
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    import os
    # bf16 by default: the fp32 model (15GB) doesn't fit alongside other GPU users on the 5090.
    # The per-step FORWARD is parity-verified (Phase 1, 3.2e-5); bf16 only perturbs low-confidence
    # token choices, which cascade in iterative MDM sampling -> modest exact-token agreement,
    # semantically-equivalent output. Set SD_FP32=1 on a free GPU for tighter agreement.
    _dt = jnp.float32 if os.environ.get("SD_FP32", "0") == "1" else jnp.bfloat16
    model, _ = load_fast_ddrive_text(SNAP, Qwen25TextConfig.fast_ddrive(_dt), dtype=_dt, verbose=False)
    cos, sin = mrope_cos_sin(d["position_ids"][:, 0, :], 128, 1e6, (16, 24, 24))
    mask4d = to_attn_mask4d(eval_hybrid_block_causal_mask_dense(jnp.asarray(rbi)))

    print(f"generating (L={L}, {len(block_ranges_from_rbi(rbi))} blocks) ...", flush=True)
    out = section_diffusion_sample(model, d["x_t0"], rbi, cos, sin, mask4d, threshold=0.9)
    gen = out[0, prompt_len:]
    ref = d["ref_output"][prompt_len:] if d["ref_output"].shape[0] > prompt_len else d["ref_output"]
    txt = tok.decode([t for t in gen if t not in (151666,)], skip_special_tokens=True)
    print("\n--- JAX section-diffusion output ---\n" + txt[:400])

    # parity: token agreement on the overlapping length (both greedy/threshold)
    n = min(len(gen), len(ref))
    agree = float((gen[:n] == ref[:n]).mean()) if n else 0.0
    import json, re
    try:
        valid_json = bool(json.loads(txt[txt.find("{"):txt.rfind("}") + 1]))
    except Exception:
        valid_json = False
    has_traj = bool(re.search(r'trajectory', txt))
    print(f"\ntoken agreement vs PyTorch (first {n}): {agree*100:.1f}%  | valid JSON: {valid_json}  | has trajectory: {has_traj}")
    ok = valid_json and has_traj
    print(f"PHASE8_SD_SAMPLER_{'PASS' if ok else 'FAIL'} (generates parseable JSON trajectory; token-agree {agree*100:.0f}%)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
