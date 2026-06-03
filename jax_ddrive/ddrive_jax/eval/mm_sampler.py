"""JAX multimodal section-diffusion sampler — the JAX port of `mdm_sample_deep_scaffold`
(mode "section_diffusion"), no-KV-cache form, with Qwen2.5-VL image fusion.

Per modeling.py/generation_utils.py:
  * ViT(pixel_values) is run ONCE → merged image embeds, scattered into the text stream
    at image-token (151655) positions; image embeds are FIXED across denoising steps.
  * Attention = the hybrid block-causal eval mask from rbi (prompt causal, response→prompt
    all, response→response block-causal). Image tokens live in the prompt (rbi=-1).
  * Per block, iterate: shifted-logit (predict pos p from logit p-1) argmax + confidence
    `>threshold` batch-unmask (fallback: single most-confident masked pos) until no MASK.

The per-step forward is the parity-verified JAX forward (Phase-1 logits 3.2e-5,
Phase-4b multimodal 7.7e-5)."""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from ddrive_jax.models.qwen2_5_text import mrope_cos_sin
from ddrive_jax.diffusion.masks import eval_hybrid_block_causal_mask_dense, to_attn_mask4d

IMAGE_TOKEN_ID = 151655
NULL_ID = 151666


def block_ranges_from_rbi(rbi):
    """Maximal contiguous spans of equal response_block_idx (>=0)."""
    rbi = np.asarray(rbi).reshape(-1)
    ranges, i, L = [], 0, len(rbi)
    while i < L:
        if rbi[i] >= 0:
            j = i
            while j < L and rbi[j] == rbi[i]:
                j += 1
            ranges.append((int(rbi[i]), i, j))
            i = j
        else:
            i += 1
    return ranges


def mm_section_diffusion_sample(text, vit, x_t0, rbi, position_ids, pixel_values, image_grid_thw,
                                *, mask_id=151665, threshold=0.9, max_tokens=512,
                                dtype=jnp.float32):
    """Returns generated x_t [L] (int64) with scaffold value slots filled.
    pixel_values/image_grid_thw=None → text-only (no ViT/fusion)."""
    x_t = np.asarray(x_t0, np.int64).reshape(-1)
    L = x_t.shape[0]
    cos, sin = mrope_cos_sin(np.asarray(position_ids), 128, 1e6, (16, 24, 24))   # [L, 128]
    mask4d = to_attn_mask4d(eval_hybrid_block_causal_mask_dense(jnp.asarray(rbi)))  # [1,1,L,L] bool

    multimodal = pixel_values is not None and image_grid_thw is not None
    if multimodal:
        img_emb = jnp.asarray(vit(jnp.asarray(pixel_values, dtype), np.asarray(image_grid_thw)), dtype)
        img_pos = np.where(x_t == IMAGE_TOKEN_ID)[0]
        assert img_pos.shape[0] == img_emb.shape[0], \
            f"#image tokens {img_pos.shape[0]} != ViT rows {img_emb.shape[0]}"
        img_pos_j = jnp.asarray(img_pos)

        @nnx.jit
        def hid_fwd(model, ids_row):
            emb = model.embed_tokens(ids_row[None])[0]               # [L, D]
            emb = emb.at[img_pos_j].set(img_emb.astype(emb.dtype))   # fixed image embeds
            return model.hidden_forward_mrope_cs(emb[None], cos, sin, mask4d)  # [1, L, D]
    else:
        @nnx.jit
        def hid_fwd(model, ids_row):
            emb = model.embed_tokens(ids_row[None])
            return model.hidden_forward_mrope_cs(emb, cos, sin, mask4d)

    steps = 0
    for (_, s, e) in block_ranges_from_rbi(rbi):
        n_mask = int((x_t[s:e] == mask_id).sum())
        for _ in range(n_mask + 5):
            cur = (x_t[s:e] == mask_id)
            if cur.sum() == 0:
                break
            hidden = hid_fwd(text, jnp.asarray(x_t))                 # [1, L, D]
            sec = np.asarray(text.attend(hidden[0, s - 1:e - 1]), np.float32)  # causal shift -> [blk, V]
            x1 = sec.argmax(-1)
            p = jax.nn.softmax(jnp.asarray(sec), -1)
            x1p = np.asarray(jnp.take_along_axis(p, jnp.asarray(x1)[:, None], -1)[:, 0])
            x1p = np.where(cur, x1p, -np.inf)
            unmask = np.where(x1p > threshold)[0]
            if unmask.size == 0:
                unmask = np.array([int(np.argmax(x1p))])
            x_t[s + unmask] = x1[unmask]
            steps += 1
            if steps > max_tokens:
                break
    return x_t


def decode_generation(x_t, orig_len, tokenizer):
    """Strip NULL/MASK from the generated region and decode (mirrors generation_utils
    post-processing)."""
    gen = [int(t) for t in np.asarray(x_t).reshape(-1)[orig_len:]]
    gen = [t for t in gen if t not in (NULL_ID, 151665)]
    return tokenizer.decode(gen, skip_special_tokens=True)
