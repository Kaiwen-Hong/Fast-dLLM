"""SASD (Section-Aware Stochastic Diffusion) math — native MaxText port.

BIT-EXACT port of the validated Fast-dDrive NNX reference at
``jax_ddrive/ddrive_jax/diffusion/{sasd_loss,noise,masks}.py``. Every jnp/np op,
axis, dtype, and ordering is preserved so results match the NNX originals for-bit.
This module is the SASD *math* only — loss + per-step noising + hybrid block-causal
mask. Wiring into MaxText's train.py / model / data lives in later phases.

Ported surfaces (NNX source -> here):
  - section_weighted_ce / causal_ce        <- diffusion/sasd_loss.py
  - make_batch / num_items                 <- diffusion/noise.py
  - hybrid_block_causal_mask_dense / to_attn_mask4d  <- diffusion/masks.py
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Constants (noise.py:11)
# ---------------------------------------------------------------------------
MASK_ID, IM_END, EPS = 151665, 151645, 1e-3


# ---------------------------------------------------------------------------
# Loss (port of sasd_loss.py)
# ---------------------------------------------------------------------------
_CE_VOCAB_TILES = 8


def _ce_per_token(logits2d, labels1d, ignore: int = -100):
    """Vocab-tiled fp32 per-token NLL; returns (nll[N] with 0 at ignored, valid_bool[N]).

    nll = logsumexp(logits) - logits[target] == -log_softmax(logits)[target], computed as an
    online logsumexp over a few static vocab tiles. Each tile is upcast to fp32 independently,
    so the loss-side fp32 peak is [N, V_tile] instead of [N, V]. This keeps the large-vocab CE
    TPU-safe regardless of whether XLA fuses a full-logits bf16->fp32 convert into the reduction.
    No jax.lax.map and no per-row jax.checkpoint: the previous row-chunked form lowered
    pathologically on TPU XLA."""
    valid = labels1d != ignore
    safe = jnp.where(valid, labels1d, 0)

    # Gather before casting so a bf16 logits matrix is not materialized as fp32 just to read [N].
    tgt = jnp.take_along_axis(logits2d, safe[:, None], axis=-1)[:, 0].astype(jnp.float32)

    V = logits2d.shape[-1]
    m = jnp.full(labels1d.shape, -jnp.inf, dtype=jnp.float32)
    s = jnp.zeros(labels1d.shape, dtype=jnp.float32)

    for tile_idx in range(_CE_VOCAB_TILES):
        start = (tile_idx * V) // _CE_VOCAB_TILES
        end = ((tile_idx + 1) * V) // _CE_VOCAB_TILES
        if start == end:
            continue
        tile = logits2d[:, start:end].astype(jnp.float32)
        tile_max = jnp.max(tile, axis=-1)
        m_new = jnp.maximum(m, tile_max)
        old_scale = jnp.where(jnp.isneginf(m), 0.0, jnp.exp(m - m_new))
        shifted = jnp.where(jnp.isneginf(m_new)[:, None], -jnp.inf, tile - m_new[:, None])
        s = s * old_scale + jnp.sum(jnp.exp(shifted), axis=-1)
        m = m_new

    nll = m + jnp.log(s) - tgt                                        # [N] == -log_softmax[target]
    return jnp.where(valid, nll, 0.0), valid


def section_weighted_ce(logits, labels, weights, num_items=None, vocab_size=None):
    """logits [B,L,V], labels [B,L] (-100 ignore), weights [B,L]. Causal shift by 1."""
    V = logits.shape[-1]
    sl = logits[:, :-1, :].reshape(-1, V)
    lab = labels[:, 1:].reshape(-1)
    w = weights[:, 1:].reshape(-1)
    nll, valid = _ce_per_token(sl, lab)
    weighted = nll * w
    if num_items is not None:
        return weighted.sum() / num_items
    return weighted.sum() / jnp.maximum(valid.sum(), 1)


def causal_ce(logits, labels, num_items=None):
    """Standard next-token CE (HF ForCausalLMLoss): shift by 1, ignore -100."""
    V = logits.shape[-1]
    sl = logits[:, :-1, :].reshape(-1, V)
    lab = labels[:, 1:].reshape(-1)
    nll, valid = _ce_per_token(sl, lab)
    if num_items is not None:
        return nll.sum() / num_items
    return nll.sum() / jnp.maximum(valid.sum(), 1)


# ---------------------------------------------------------------------------
# Noising (port of noise.py)
# ---------------------------------------------------------------------------
def make_batch(s: dict, rng: np.random.Generator, *, fixed_mask: np.ndarray | None = None):
    """s: one sample's arrays from prep (input_ids, labels, rbi, scaffold, weight_vec,
    block_alpha, block_beta). Returns numpy (input_final[2,2L], labels_final[2,L],
    original_labels[1,L], weights[2,L]).  fixed_mask: optional [L] bool for deterministic eval."""
    ids = s["input_ids"]; labels = s["labels"]; rbi = s["rbi"]; scaff = s["scaffold"]
    L = ids.shape[0]
    resp = labels != -100

    if fixed_mask is not None:
        mask_indices = fixed_mask.copy()
    else:
        t = rng.beta(s["block_alpha"], s["block_beta"])            # [n_blocks]
        p = (1 - EPS) * t + EPS
        pmask = np.zeros(L, np.float32)
        valid = rbi >= 0
        pmask[valid] = p[rbi[valid]]
        mask_indices = (rng.random(L) < pmask) & resp & (~scaff)
    mask_indices = mask_indices | ((ids == IM_END) & resp)         # always_mask_im_end

    noisy = ids.copy(); noisy[mask_indices] = MASK_ID
    lab_m = labels.copy(); lab_m[~mask_indices] = -100
    doubled = np.concatenate([noisy, ids])                         # [2L]

    comp = (resp & ~mask_indices) | ((ids == IM_END) & resp)
    comp = comp & (~scaff)
    cnoisy = ids.copy(); cnoisy[comp] = MASK_ID
    clab = labels.copy(); clab[~comp] = -100
    cdoubled = np.concatenate([cnoisy, ids])

    input_final = np.stack([doubled, cdoubled]).astype(np.int64)   # [2, 2L]
    labels_final = np.stack([lab_m, clab]).astype(np.int64)        # [2, L]
    original_labels = labels[None].astype(np.int64)                # [1, L]
    weights = np.stack([s["weight_vec"], s["weight_vec"]]).astype(np.float32)  # [2, L]
    return input_final, labels_final, original_labels, weights


def num_items(s: dict) -> float:
    return float(2 * int((s["labels"] != -100).sum()))


# ---------------------------------------------------------------------------
# Hybrid block-causal mask (port of masks.py)
# ---------------------------------------------------------------------------
def hybrid_block_causal_mask_dense(response_block_idx, turn_idx, n: int):
    """Dense [2n, 2n] bool mask for the doubled training sequence.

    response_block_idx, turn_idx: int arrays [n] (per original position; -1 = prompt).
    True = may attend.
    """
    idx = jnp.arange(2 * n)
    x0 = idx >= n                                   # [2n] clean-half flag
    pos = jnp.where(x0, idx - n, idx)               # [2n] original position
    block = response_block_idx[pos]                 # [2n]
    turn = turn_idx[pos]                            # [2n]

    x0_q, x0_kv = x0[:, None], x0[None, :]
    pos_q, pos_kv = pos[:, None], pos[None, :]
    turn_q, turn_kv = turn[:, None], turn[None, :]

    block_diagonal = (~x0_q) & (~x0_kv) & (turn_q == turn_kv)
    offset_block_causal = (turn_q > turn_kv) & x0_kv & (~x0_q)
    x0_causal = x0_q & x0_kv & (pos_q >= pos_kv)
    return block_diagonal | offset_block_causal | x0_causal      # [2n, 2n] bool


def to_attn_mask4d(mask2d):
    """[Q, K] bool -> [1, 1, Q, K] bool for attention (True = attend)."""
    return mask2d[None, None, :, :]


# ---------------------------------------------------------------------------
# 3D M-RoPE (port of ddrive_jax.models.qwen2_5_text.mrope_cos_sin)
# ---------------------------------------------------------------------------
# CRITICAL: this is the CONTIGUOUS-CHUNK layout ([T..H..W..]) the Fast-dDrive NNX
# reference uses. It is NOT the same as MaxText's built-in use_mrope path
# (Qwen3OmniMoeThinkerTextRotaryEmbedding._apply_interleaved_mrope), which is an
# INTERLEAVED scatter ([THTW..]). The base inv_freq / timescale match between the
# two (theta**(-2i/hd), min_timescale=1, max_timescale=theta); only the section
# combination differs. For SASD we therefore compute cos/sin here (bit-identical
# to NNX) and inject them, bypassing MaxText RoPE entirely (config.use_mrope MUST
# be false). See README + the sasd train branch.
def mrope_cos_sin(pos3d, head_dim: int, theta: float, mrope_section):
    """3D M-RoPE combined cos/sin [L, head_dim] from position_ids [3, L].

    Bit-exact port of ddrive_jax.models.qwen2_5_text.mrope_cos_sin (every np op,
    float64 accumulation, contiguous slice order preserved).
    """
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))  # [hd/2]
    pos = np.asarray(pos3d, np.float64)                                              # [3, L]
    freqs = pos[:, :, None] * inv[None, None, :]                                     # [3, L, hd/2]
    emb = np.concatenate([freqs, freqs], -1)                                         # [3, L, hd]
    cos, sin = np.cos(emb), np.sin(emb)
    sec = list(mrope_section) * 2
    cparts, sparts, idx = [], [], 0
    for i, s in enumerate(sec):
        cparts.append(cos[i % 3, :, idx:idx + s])
        sparts.append(sin[i % 3, :, idx:idx + s])
        idx += s
    return (jnp.asarray(np.concatenate(cparts, -1)),
            jnp.asarray(np.concatenate(sparts, -1)))  # [L, hd] each


def _rotate_half(x):
    """(x1, x2) -> (-x2, x1) on the last axis (HF / MaxText full-form rope)."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return jnp.concatenate([-x2, x1], -1)


def apply_rope_full(x, cos, sin):
    """HF full-form rope, fp32. x [B,L,N,Dh]; cos,sin [B,L,Dh] (or broadcastable).

    Bit-exact match to ddrive_jax.models.qwen2_5_text.apply_rope_full:
        (x*cos + rotate_half(x)*sin) in fp32, cast back to x.dtype.
    cos/sin get a head axis inserted so they broadcast over N.
    """
    cos = cos[:, :, None, :].astype(jnp.float32)
    sin = sin[:, :, None, :].astype(jnp.float32)
    xf = x.astype(jnp.float32)
    return (xf * cos + _rotate_half(xf) * sin).astype(x.dtype)


# ---------------------------------------------------------------------------
# Frozen-ViT image embeds (mirror train_tpu._real_image_embeds_fn EXACTLY)
# ---------------------------------------------------------------------------
IMAGE_TOK = 151655  # Qwen2.5-VL <|image_pad|> placeholder token id (NNX train_tpu.IMAGE_TOK)


def compute_fast_ddrive_image_embeds(batch: dict, vit, dtype=jnp.float32):
  """Frozen ViT image embeds, doubled to [B, 2N, D].

  BIT-EXACT mirror of ``ddrive_jax.train.train_tpu._real_image_embeds_fn.fn``:
  per sample b, run the (frozen) ViT on ``pixel_values[b]`` + ``image_grid_thw[b]``
  -> [N, D]; double via ``jnp.concatenate([ie, ie], 0)`` -> [2N, D] (first N rows ==
  the noisy-half image positions, second N == clean-half), ``stop_gradient`` (frozen),
  then stack -> [B, 2N, D]. ``vit`` is a loaded ddrive_jax VisionModel (the SAME object
  / weights NNX uses). dtype must match the NNX harness dtype (fp32 for parity).

  Returns a [B, 2N, D] jnp array.
  """
  B = int(batch["input_final"].shape[0])
  outs = []
  for b in range(B):
    ie = vit(jnp.asarray(batch["pixel_values"][b], dtype),
             np.asarray(batch["image_grid_thw"][b]))      # [N, D]
    ie = jnp.concatenate([ie, ie], 0).astype(dtype)        # [2N, D] (doubled sequence)
    outs.append(jax.lax.stop_gradient(ie))
  return jnp.stack(outs, 0)                                 # [B, 2N, D]


def _image_positions(batch: dict, B: int):
  """Per-sample image-token positions in the DOUBLED [2L] sequence (row-0 derived).

  Mirrors train_tpu.prepare_batch:130 ``np.where(input_final[b,0] == IMAGE_TOK)``.
  Image tokens live in the prompt (labels == -100) so they are NEVER masked; row-0 and
  row-1 carry IMAGE_TOK at identical positions, so the row-0-derived positions are valid
  for both doubled rows (exactly what NNX does). Returns ([B, 2N] i32, twoN python int).
  """
  pos_list = [np.where(batch["input_final"][b, 0] == IMAGE_TOK)[0] for b in range(B)]
  twoN = pos_list[0].shape[0]
  assert all(p.shape[0] == twoN for p in pos_list), (
      f"non-uniform image-token count per sample: {[p.shape[0] for p in pos_list]} "
      "(uniform N required for a static jit shape; matches train_tpu.prepare_batch:135)")
  return np.stack(pos_list).astype(np.int32), twoN


# ---------------------------------------------------------------------------
# Host-side SASD batch preparation (doubled cos/sin, 4D mask, flattened rows)
# ---------------------------------------------------------------------------
def prepare_sasd_inputs(batch: dict, head_dim: int, mrope_section, theta: float,
                        image_embeds=None):
    """Turn a grain SASD batch dict into the tensors the MaxText sasd branch needs.

    Mirrors ddrive_jax.train.train_tpu.prepare_batch (the doubled cos/sin + hybrid
    mask precompute, PLUS the frozen-ViT image-embed scatter inputs) laid out for
    MaxText's [batch, len] decoder via the row-flatten convention flat = 2*b + r
    (np.repeat(..., 2, axis=0)).

    ``image_embeds`` (optional [B, 2N, D] from compute_fast_ddrive_image_embeds):
    if given, also emits ``image_embeds`` ([2B, 2N, D], repeat x2) and ``img_pos``
    ([2B, 2N] i32, repeat x2) so the SASD decoder can scatter the frozen ViT embeds
    at the IMAGE_TOK positions in BOTH doubled rows -- exactly NNX's per-row
    ``embed_tokens(ifn[r]).at[img_pos].set(image_embeds)`` (same img_pos / embeds for
    r=0 and r=1). If None, those keys are omitted (text-only SASD, e.g. the tiny test).

    Input batch (numpy) keys used:
      input_final [B,2,2L] i64, labels_final [B,2,L] i64, original_labels [B,1,L] i64,
      weights [B,2,L] f32, position_ids [B,3,L] i32, rbi [B,L] i32, turn [B,L] i32,
      num_items [B] f32.

    Returns a dict of jnp arrays:
      inputs        [2B, 2L] i32   (row-flattened doubled token ids; flat = 2b+r)
      cos, sin      [2B, 2L, hd]   (NNX contiguous-chunk M-RoPE, doubled positions, repeat x2)
      attn_mask     [2B, 2L, 2L]   bool (hybrid block-causal, True=attend, repeat x2)
      labels_final  [B, 2, L] i64
      original_labels [B, 1, L] i64
      weights       [B, 2, L] f32
      num_items     [B] f32
      B, L          python ints
    """
    B = int(batch["input_final"].shape[0])
    L = int(batch["rbi"].shape[1])

    cos_list, sin_list, mask_list = [], [], []
    for b in range(B):
        pos2 = np.concatenate([batch["position_ids"][b], batch["position_ids"][b]], axis=1)  # [3,2L]
        cos_b, sin_b = mrope_cos_sin(pos2, head_dim, theta, mrope_section)                    # [2L,hd]
        mask_b = hybrid_block_causal_mask_dense(
            jnp.asarray(batch["rbi"][b]), jnp.asarray(batch["turn"][b]), L)                   # [2L,2L] bool
        cos_list.append(np.asarray(cos_b))
        sin_list.append(np.asarray(sin_b))
        mask_list.append(np.asarray(mask_b))

    cos = np.stack(cos_list)                       # [B, 2L, hd]
    sin = np.stack(sin_list)                       # [B, 2L, hd]
    mask = np.stack(mask_list)                     # [B, 2L, 2L] bool

    # row-flatten: each sample's cos/sin/mask is shared by BOTH rows -> repeat x2 to [2B,...]
    cos2 = np.repeat(cos, 2, axis=0)               # [2B, 2L, hd]
    sin2 = np.repeat(sin, 2, axis=0)
    mask2 = np.repeat(mask, 2, axis=0)             # [2B, 2L, 2L]
    inputs = batch["input_final"].reshape(2 * B, 2 * L).astype(np.int32)  # flat = 2b+r

    out = {
        "inputs": jnp.asarray(inputs),
        "cos": jnp.asarray(cos2),
        "sin": jnp.asarray(sin2),
        "attn_mask": jnp.asarray(mask2),
        "labels_final": jnp.asarray(batch["labels_final"]),
        "original_labels": jnp.asarray(batch["original_labels"]),
        "weights": jnp.asarray(batch["weights"].astype(np.float32)),
        "num_items": jnp.asarray(np.asarray(batch["num_items"]).astype(np.float32)),
        "B": B,
        "L": L,
    }

    if image_embeds is not None:
      # row-flatten the image embeds + IMAGE_TOK positions the SAME way as cos/sin/mask
      # (each sample's embeds + positions are shared by BOTH doubled rows -> repeat x2).
      img_pos, twoN = _image_positions(batch, B)                  # [B, 2N], int
      ie = np.asarray(image_embeds)                               # [B, 2N, D]
      assert ie.shape[1] == twoN, (
          f"img_pos 2N={twoN} != image_embeds 2N={ie.shape[1]} "
          "(matches train_tpu.prepare_batch:143)")
      out["image_embeds"] = jnp.asarray(np.repeat(ie, 2, axis=0))           # [2B, 2N, D]
      out["img_pos"] = jnp.asarray(np.repeat(img_pos, 2, axis=0))           # [2B, 2N] i32

    return out


def sasd_loss_from_logits(logits, labels_final, original_labels, weights, num_items, B, L):
    """Compute the SASD global loss numerator/denominator from full [2B, 2L, V] logits.

    Reproduces ddrive_jax.train.train_tpu.per_sample_loss_sums (vmapped over B) +
    the global DP-correct normaliser sum_b(sec+causal)/sum_b(num_items):
      - reshape logits [2B,2L,V] -> [B,2,2L,V] (flat=2b+r)
      - noisy half = lg[:, :, :L, :] (BOTH rows) -> [2B, L, V]
      - clean half = lg[:, 0, L:, :] (mdm row only) -> [B, L, V]
      - sec_sum   = section_weighted_ce(noisy, labels_final, weights, num_items=1.0)
      - cau_sum   = causal_ce(clean, original_labels[:,0,:], num_items=1.0)
      - loss      = (sec_sum + cau_sum) / sum_b(num_items)

    Returns (loss, sec_sum, cau_sum, total_weights). total_weights = sum_b(num_items).
    """
    V = logits.shape[-1]
    lg = logits.reshape(B, 2, 2 * L, V)
    noisy_logits = lg[:, :, :L, :].reshape(2 * B, L, V)   # both rows
    clean_logits = lg[:, 0, L:, :]                        # row-0 (mdm) only -> [B, L, V]

    sec_sum = section_weighted_ce(
        noisy_logits, labels_final.reshape(2 * B, L), weights.reshape(2 * B, L), num_items=1.0)
    cau_sum = causal_ce(clean_logits, original_labels[:, 0, :], num_items=1.0)

    total_weights = jnp.sum(num_items)
    loss = (sec_sum + cau_sum) / jnp.maximum(total_weights, 1.0)
    return loss, sec_sum, cau_sum, total_weights


__all__ = [
    "MASK_ID", "IM_END", "EPS",
    "section_weighted_ce", "causal_ce",
    "make_batch", "num_items",
    "hybrid_block_causal_mask_dense", "to_attn_mask4d",
    "mrope_cos_sin", "apply_rope_full",
    "prepare_sasd_inputs", "sasd_loss_from_logits",
    "IMAGE_TOK", "compute_fast_ddrive_image_embeds",
]
