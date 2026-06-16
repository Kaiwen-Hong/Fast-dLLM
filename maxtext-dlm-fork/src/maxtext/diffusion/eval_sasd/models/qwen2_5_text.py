# ── VENDORED from Fast-dLLM ddrive_jax/models/qwen2_5_text.py @ 4b0f4f2 — DO NOT EDIT logic here; sync from source.
# Self-contained SASD inference for the internal-TPU fork (B2). See ../PATCHES.md / docs.
"""Qwen2.5 text decoder (Fast-dDrive LM) in Flax NNX.

Adapted from jax-mdlm-handoff/code/models/qwen3.py. Fast-dDrive's text tower is the
Qwen2.5-3B family: GQA, qkv BIAS, NO q/k-norm, tied embeddings, RoPE theta 1e6,
RMSNorm eps 1e-6, SiLU MLP. Matches PyTorch eager_attention_forward exactly:
  attn = softmax( (q·kᵀ)*head_dim**-0.5 + additive_mask )  (fp32 softmax)
RoPE = HF rotate-half. M-RoPE is a no-op here (text-only positions = arange).

The module accepts an explicit boolean attention mask (True = attend) and explicit
position_ids, so the same backbone serves Phase-1 parity (bidirectional), Phase-2
hybrid block-causal diffusion, and standard causal use.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, DTypeLike

from .rope import RoPE, apply_rope

LARGE_NEGATIVE = jnp.finfo(jnp.float32).min


@dataclass
class Qwen25TextConfig:
    dtype: DTypeLike = jnp.float32
    d_model: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 2
    head_dim: int = 128
    n_layers: int = 36
    mlp_hidden_size: int = 11008
    vocab_size: int = 151936
    max_sequence_length: int = 128000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    include_qkv_bias: bool = True     # Qwen2.5: q/k/v have bias
    include_bias: bool = False        # o_proj / mlp: no bias
    use_q_k_norm: bool = False        # Qwen2.5: no q/k norm (Qwen3-only)
    tie_embeddings: bool = True
    return_hidden_states: bool = False

    @staticmethod
    def fast_ddrive(dtype: DTypeLike = jnp.float32) -> "Qwen25TextConfig":
        """Matches Efficient-Large-Model/Fast-dDrive config.json text_config."""
        return Qwen25TextConfig(
            dtype=dtype, d_model=2048, n_heads=16, n_kv_heads=2, head_dim=128,
            n_layers=36, mlp_hidden_size=11008, vocab_size=151936,
            rms_norm_eps=1e-6, rope_theta=1_000_000.0,
            include_qkv_bias=True, include_bias=False, use_q_k_norm=False,
            tie_embeddings=True,
        )


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float, *, dtype: DTypeLike, rngs: nnx.Rngs):
        self.weight = nnx.Param(jnp.ones((dim,), dtype=dtype))
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        dt = x.dtype
        x32 = x.astype(jnp.float32)
        norm = x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + self.eps)
        return (norm * self.weight[...].astype(jnp.float32)).astype(dt)


class Linear(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, *, use_bias: bool, dtype: DTypeLike, rngs: nnx.Rngs):
        self.kernel = nnx.Param(jax.nn.initializers.lecun_normal()(rngs.params(), (in_dim, out_dim), dtype=dtype))
        self.bias = nnx.Param(jnp.zeros((out_dim,), dtype=dtype)) if use_bias else None

    def __call__(self, x: Array) -> Array:
        y = jnp.matmul(x, self.kernel)
        if self.bias is not None:
            y = y + self.bias
        return y


class Qwen25Attention(nnx.Module):
    def __init__(self, cfg: Qwen25TextConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        q_dim, kv_dim = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
        self.q_proj = Linear(cfg.d_model, q_dim, use_bias=cfg.include_qkv_bias, dtype=cfg.dtype, rngs=rngs)
        self.k_proj = Linear(cfg.d_model, kv_dim, use_bias=cfg.include_qkv_bias, dtype=cfg.dtype, rngs=rngs)
        self.v_proj = Linear(cfg.d_model, kv_dim, use_bias=cfg.include_qkv_bias, dtype=cfg.dtype, rngs=rngs)
        self.o_proj = Linear(q_dim, cfg.d_model, use_bias=cfg.include_bias, dtype=cfg.dtype, rngs=rngs)
        if cfg.use_q_k_norm:
            self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps, dtype=cfg.dtype, rngs=rngs)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps, dtype=cfg.dtype, rngs=rngs)
        else:
            self.q_norm = self.k_norm = None

    def __call__(self, x: Array, sin: Array, cos: Array, mask4d: Array | None,
                 mrope_cs=None) -> Array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if mrope_cs is not None:                                  # 3D M-RoPE (multimodal)
            cosf, sinf = mrope_cs
            q, k = apply_rope_full(q, cosf, sinf), apply_rope_full(k, cosf, sinf)
        else:                                                    # plain RoPE (text)
            q, k = apply_rope(q, sin, cos), apply_rope(k, sin, cos)
        if self.n_rep > 1:
            k = jnp.repeat(k, self.n_rep, axis=2)
            v = jnp.repeat(v, self.n_rep, axis=2)
        scale = jnp.asarray(self.head_dim ** -0.5, dtype=jnp.float32)
        logits = jnp.einsum("BTNH,BSNH->BNTS", q, k).astype(jnp.float32) * scale
        if mask4d is not None:
            # mask4d: bool [B|1, 1, T, S], True = attend.
            logits = jnp.where(mask4d, logits, LARGE_NEGATIVE)
        probs = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        out = jnp.einsum("BNTS,BSNH->BTNH", probs, v).reshape(B, L, -1)
        return self.o_proj(out)


class Qwen25MLP(nnx.Module):
    def __init__(self, cfg: Qwen25TextConfig, *, rngs: nnx.Rngs):
        self.gate_proj = Linear(cfg.d_model, cfg.mlp_hidden_size, use_bias=cfg.include_bias, dtype=cfg.dtype, rngs=rngs)
        self.up_proj = Linear(cfg.d_model, cfg.mlp_hidden_size, use_bias=cfg.include_bias, dtype=cfg.dtype, rngs=rngs)
        self.down_proj = Linear(cfg.mlp_hidden_size, cfg.d_model, use_bias=cfg.include_bias, dtype=cfg.dtype, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


def _rotate_half_full(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return jnp.concatenate([-x2, x1], -1)


def apply_rope_full(x, cos, sin):
    """HF full-form rope: x [B,L,H,Dh]; cos,sin [L,Dh]. Used for (multimodal) M-RoPE.
    Equivalent to the split-form when cos=sin pattern is cat(c,c)."""
    cos = cos[None, :, None, :].astype(jnp.float32)
    sin = sin[None, :, None, :].astype(jnp.float32)
    xf = x.astype(jnp.float32)
    return (xf * cos + _rotate_half_full(xf) * sin).astype(x.dtype)


def mrope_cos_sin(pos3d, head_dim: int, theta: float, mrope_section):
    """3D M-RoPE combined cos/sin [L, head_dim] from position_ids [3, L]
    (port of apply_multimodal_rotary_pos_emb + Fast_dDriveRotaryEmbedding)."""
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))   # [hd/2]
    pos = np.asarray(pos3d, np.float64)                                               # [3, L]
    freqs = pos[:, :, None] * inv[None, None, :]                                       # [3, L, hd/2]
    emb = np.concatenate([freqs, freqs], -1)                                           # [3, L, hd]
    cos, sin = np.cos(emb), np.sin(emb)
    sec = list(mrope_section) * 2
    cparts, sparts, idx = [], [], 0
    for i, s in enumerate(sec):
        cparts.append(cos[i % 3, :, idx:idx + s]); sparts.append(sin[i % 3, :, idx:idx + s]); idx += s
    return jnp.asarray(np.concatenate(cparts, -1)), jnp.asarray(np.concatenate(sparts, -1))  # [L, hd] each


def _layer_call(layer, x, sin, cos, mask4d):
    return layer(x, sin, cos, mask4d)


def _layer_call_mrope(layer, x, mask4d, cos, sin):
    return layer(x, None, None, mask4d, (cos, sin))


class Qwen25DecoderLayer(nnx.Module):
    def __init__(self, cfg: Qwen25TextConfig, *, rngs: nnx.Rngs):
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.rms_norm_eps, dtype=cfg.dtype, rngs=rngs)
        self.self_attn = Qwen25Attention(cfg, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.rms_norm_eps, dtype=cfg.dtype, rngs=rngs)
        self.mlp = Qwen25MLP(cfg, rngs=rngs)

    def __call__(self, x: Array, sin: Array, cos: Array, mask4d: Array | None,
                 mrope_cs=None) -> Array:
        x = x + self.self_attn(self.input_layernorm(x), sin, cos, mask4d, mrope_cs)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen25TextModel(nnx.Module):
    def __init__(self, cfg: Qwen25TextConfig, *, rngs: nnx.Rngs):
        self.config = cfg
        self.embed_tokens = nnx.Embed(cfg.vocab_size, cfg.d_model, dtype=cfg.dtype, rngs=rngs)
        self.layers = nnx.List([Qwen25DecoderLayer(cfg, rngs=rngs) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps, dtype=cfg.dtype, rngs=rngs)
        if not cfg.tie_embeddings:
            self.lm_head = Linear(cfg.d_model, cfg.vocab_size, use_bias=False, dtype=cfg.dtype, rngs=rngs)
        self.rope = RoPE(rope_type="default", head_dim=cfg.head_dim, rope_theta=cfg.rope_theta)

    def attend(self, hidden: Array) -> Array:
        return self.embed_tokens.attend(hidden) if self.config.tie_embeddings else self.lm_head(hidden)

    def hidden_forward_mrope(self, inputs_embeds: Array, position_ids_3d, mask4d: Array | None,
                             mrope_section=(16, 24, 24)) -> Array:
        """Multimodal forward from pre-fused embeds with 3D M-RoPE. position_ids_3d: [3, L].
        (Not jit-able: mrope_cos_sin is host-side numpy. For training use the _cs variant.)"""
        cosf, sinf = mrope_cos_sin(position_ids_3d, self.config.head_dim,
                                   self.config.rope_theta, mrope_section)
        return self.hidden_forward_mrope_cs(inputs_embeds, cosf, sinf, mask4d)

    def hidden_forward_mrope_cs(self, inputs_embeds: Array, cos: Array, sin: Array,
                                mask4d: Array | None, *, remat: bool = False) -> Array:
        """M-RoPE forward with PRECOMPUTED combined cos/sin [L, head_dim] (jit-able)."""
        x = inputs_embeds
        call = nnx.remat(_layer_call_mrope) if remat else _layer_call_mrope
        for layer in self.layers:
            x = call(layer, x, mask4d, cos, sin)
        return self.norm(x)

    def hidden_forward(self, input_ids: Array, mask4d: Array | None = None,
                       position_ids: Array | None = None, *, remat: bool = False) -> Array:
        """Final post-norm hidden [B, L, D] without lm_head (slice + attend only what
        you need to avoid a [B*L, V] matmul / OOM on long sequences). remat=True turns on
        per-layer gradient checkpointing for training under tight GPU memory."""
        return self.forward_with_embeds(self.embed_tokens(input_ids), mask4d, position_ids,
                                        logits=False, remat=remat)

    def __call__(self, input_ids: Array, mask4d: Array | None = None,
                 position_ids: Array | None = None):
        x = self.embed_tokens(input_ids)
        return self.forward_with_embeds(x, mask4d, position_ids)

    def forward_with_embeds(self, x: Array, mask4d: Array | None = None,
                            position_ids: Array | None = None, *, logits: bool = True,
                            remat: bool = False):
        B, L, _ = x.shape
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(L, dtype=jnp.int32)[None, :], (B, L))
        sin, cos = self.rope(position_ids)
        call = nnx.remat(_layer_call) if remat else _layer_call
        hidden_states: list[Array] = []
        for layer in self.layers:
            if self.config.return_hidden_states:
                hidden_states.append(x)
            x = call(layer, x, sin, cos, mask4d)
        x = self.norm(x)
        if not logits:
            return x
        if self.config.return_hidden_states:
            hidden_states.append(x)
        return self.attend(x), hidden_states
