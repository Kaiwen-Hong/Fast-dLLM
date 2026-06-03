"""Capture a PyTorch text-only oracle forward for Phase-1 parity.

Bypasses the diffusion wrapper: calls model.model.language_model directly with an
explicit bidirectional (all-zeros additive) 4D mask and explicit arange position_ids,
forcing EAGER attention so the math matches the JAX softmax exactly. Saves
input_ids, post-norm hidden, and tied-lm_head logits to an npz.
"""
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
OUT = "/home/kaiwen/data/fast-ddrive/ref_logits/text_oracle.npz"
L = 48
SEED = 0


def main():
    torch.manual_seed(SEED)
    print("loading Fast-dDrive (fp32, eager attention) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.float32, trust_remote_code=True,
        attn_implementation="eager",
    ).eval().cuda()

    vocab = model.config.text_config.vocab_size
    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, vocab, size=(1, L)).astype(np.int64)
    ids_t = torch.tensor(ids, device="cuda")

    lm = model.model.language_model
    emb = lm.embed_tokens(ids_t)
    pos = torch.arange(L, device="cuda")[None, :]                 # [1, L] -> expands to [3,1,L]
    mask = torch.zeros(1, 1, L, L, device="cuda", dtype=torch.float32)  # additive 0 => bidirectional

    with torch.no_grad():
        out = lm(inputs_embeds=emb, attention_mask=mask, position_ids=pos, use_cache=False)
        hidden = out.last_hidden_state                            # already RMSNorm'd
        logits = model.lm_head(hidden)                            # tied head

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT,
             input_ids=ids,
             hidden=hidden.float().cpu().numpy(),
             logits=logits.float().cpu().numpy())
    print(f"saved {OUT}")
    print("logits", tuple(logits.shape), "abs-max", float(logits.abs().max()),
          "hidden abs-max", float(hidden.abs().max()))
    print("ORACLE_DONE")


if __name__ == "__main__":
    main()
