"""Qwen2.5-VL vision tower (Fast-dDrive ViT) in Flax NNX.

Faithful port of modeling.py:307-735. Index logic (2D-RoPE positions, window
partitioning, segment boundaries) is computed host-side in numpy as exact ports of
`rot_pos_emb` / `get_window_index`; the neural ops (patch embed as a linear, RMSNorm,
2D-rotary attention with per-segment masking, SwiGLU MLP, patch merger) run in NNX.

Config (config.json vision_config): depth 32, hidden 1280, heads 16, patch 14,
temporal_patch 2, spatial_merge 2, window 112, full-attn blocks [7,15,23,31],
out_hidden 2048, intermediate 3420, RMSNorm eps 1e-6, SiLU. ViT-RoPE theta 10000.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, DTypeLike

LARGE_NEGATIVE = jnp.finfo(jnp.float32).min


@dataclass
class VisionConfig:
    dtype: DTypeLike = jnp.float32
    depth: int = 32
    hidden_size: int = 1280
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    window_size: int = 112
    fullatt_block_indexes: tuple = (7, 15, 23, 31)
    out_hidden_size: int = 2048
    intermediate_size: int = 3420
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads

    @property
    def spatial_merge_unit(self):
        return self.spatial_merge_size ** 2


# ---------------- host-side index ports (numpy) ----------------

def rot_pos_ids(grid_thw, spatial_merge_size: int) -> np.ndarray:
    """Exact port of rot_pos_emb's pos_ids construction (modeling.py:607-630). -> [N,2] (h,w)."""
    out = []
    s = spatial_merge_size
    for t, h, w in grid_thw:
        hpos = np.arange(h)[:, None].repeat(w, 1)
        hpos = hpos.reshape(h // s, s, w // s, s).transpose(0, 2, 1, 3).reshape(-1)
        wpos = np.arange(w)[None, :].repeat(h, 0)
        wpos = wpos.reshape(h // s, s, w // s, s).transpose(0, 2, 1, 3).reshape(-1)
        out.append(np.stack([hpos, wpos], -1).repeat(t, 0) if t > 1 else np.stack([hpos, wpos], -1))
    return np.concatenate(out, 0)


def get_window_index(grid_thw, cfg: VisionConfig):
    """Exact port of get_window_index (modeling.py:636-675). -> (window_index[N//u], cu_window_seqlens)."""
    s, u = cfg.spatial_merge_size, cfg.spatial_merge_unit
    win = cfg.window_size // s // cfg.patch_size
    window_index, cu = [], [0]
    wid = 0
    for t, h, w in grid_thw:
        lh, lw = h // s, w // s
        idx = np.arange(t * lh * lw).reshape(t, lh, lw)
        ph = (win - lh % win) % win
        pw = (win - lw % win) % win
        nh, nw = (lh + ph) // win, (lw + pw) // win
        idxp = np.pad(idx, ((0, 0), (0, ph), (0, pw)), constant_values=-100)
        idxp = idxp.reshape(t, nh, win, nw, win).transpose(0, 1, 3, 2, 4).reshape(t, nh * nw, win, win)
        seqlens = (idxp != -100).sum((2, 3)).reshape(-1)
        idxp = idxp.reshape(-1)
        window_index.append(idxp[idxp != -100] + wid)
        cu_tmp = np.cumsum(seqlens) * u + cu[-1]
        cu.extend(cu_tmp.tolist())
        wid += t * lh * lw
    return np.concatenate(window_index), np.array(cu, np.int64)


def cu_seqlens_full(grid_thw) -> np.ndarray:
    """Full-attention boundaries (modeling.py:708-716)."""
    reps = np.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return np.concatenate([[0], np.cumsum(reps)]).astype(np.int64)


def _seg_ids_from_cu(cu: np.ndarray, n: int) -> np.ndarray:
    """Segment id per position from cumulative seqlens (unique-consecutive applied)."""
    cu = np.unique(cu)
    seg = np.zeros(n, np.int32)
    for i in range(len(cu) - 1):
        seg[cu[i]:cu[i + 1]] = i
    return seg


# ---------------- NNX modules ----------------

class RMSNorm(nnx.Module):
    def __init__(self, dim, eps, *, dtype, rngs):
        self.weight = nnx.Param(jnp.ones((dim,), dtype=dtype)); self.eps = eps

    def __call__(self, x):
        dt = x.dtype; x32 = x.astype(jnp.float32)
        n = x32 * jax.lax.rsqrt(jnp.mean(x32 ** 2, -1, keepdims=True) + self.eps)
        return (n * self.weight[...].astype(jnp.float32)).astype(dt)


class Lin(nnx.Module):
    def __init__(self, i, o, *, bias, dtype, rngs):
        self.kernel = nnx.Param(jax.nn.initializers.lecun_normal()(rngs.params(), (i, o), dtype=dtype))
        self.bias = nnx.Param(jnp.zeros((o,), dtype=dtype)) if bias else None

    def __call__(self, x):
        y = jnp.matmul(x, self.kernel)
        return y + self.bias if self.bias is not None else y


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return jnp.concatenate([-x2, x1], -1)


def _apply_vision_rope(q, k, cos, sin):
    # q,k: [N, heads, hd]; cos,sin: [N, hd]
    cos = cos[:, None, :].astype(jnp.float32); sin = sin[:, None, :].astype(jnp.float32)
    qf, kf = q.astype(jnp.float32), k.astype(jnp.float32)
    qe = qf * cos + _rotate_half(qf) * sin
    ke = kf * cos + _rotate_half(kf) * sin
    return qe.astype(q.dtype), ke.astype(k.dtype)


class VisionAttention(nnx.Module):
    def __init__(self, cfg, *, rngs):
        self.h = cfg.num_heads; self.hd = cfg.head_dim
        self.qkv = Lin(cfg.hidden_size, cfg.hidden_size * 3, bias=True, dtype=cfg.dtype, rngs=rngs)
        self.proj = Lin(cfg.hidden_size, cfg.hidden_size, bias=True, dtype=cfg.dtype, rngs=rngs)
        self.scale = cfg.head_dim ** -0.5

    def __call__(self, x, cos, sin, seg_mask):
        N, _ = x.shape
        qkv = self.qkv(x).reshape(N, 3, self.h, self.hd).transpose(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # [N, h, hd]
        q, k = _apply_vision_rope(q, k, cos, sin)
        scores = jnp.einsum("nhd,mhd->hnm", q, k).astype(jnp.float32) * self.scale
        scores = jnp.where(seg_mask[None], scores, LARGE_NEGATIVE)  # seg_mask [N,N]
        p = jax.nn.softmax(scores, -1).astype(v.dtype)
        out = jnp.einsum("hnm,mhd->nhd", p, v).reshape(N, -1)
        return self.proj(out)


class VisionMLP(nnx.Module):
    def __init__(self, cfg, *, rngs):
        self.gate_proj = Lin(cfg.hidden_size, cfg.intermediate_size, bias=True, dtype=cfg.dtype, rngs=rngs)
        self.up_proj = Lin(cfg.hidden_size, cfg.intermediate_size, bias=True, dtype=cfg.dtype, rngs=rngs)
        self.down_proj = Lin(cfg.intermediate_size, cfg.hidden_size, bias=True, dtype=cfg.dtype, rngs=rngs)

    def __call__(self, x):
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class VisionBlock(nnx.Module):
    def __init__(self, cfg, *, rngs):
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=cfg.dtype, rngs=rngs)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=cfg.dtype, rngs=rngs)
        self.attn = VisionAttention(cfg, rngs=rngs)
        self.mlp = VisionMLP(cfg, rngs=rngs)

    def __call__(self, x, cos, sin, seg_mask):
        x = x + self.attn(self.norm1(x), cos, sin, seg_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerger(nnx.Module):
    def __init__(self, cfg, *, rngs):
        self.hidden = cfg.hidden_size * cfg.spatial_merge_unit
        self.ln_q = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=cfg.dtype, rngs=rngs)
        self.fc1 = Lin(self.hidden, self.hidden, bias=True, dtype=cfg.dtype, rngs=rngs)
        self.fc2 = Lin(self.hidden, cfg.out_hidden_size, bias=True, dtype=cfg.dtype, rngs=rngs)

    def __call__(self, x):
        x = self.ln_q(x).reshape(-1, self.hidden)
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=False))


class VisionTransformer(nnx.Module):
    def __init__(self, cfg: VisionConfig, *, rngs):
        self.cfg = cfg
        patch_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        self.patch_embed = Lin(patch_dim, cfg.hidden_size, bias=False, dtype=cfg.dtype, rngs=rngs)
        self.blocks = nnx.List([VisionBlock(cfg, rngs=rngs) for _ in range(cfg.depth)])
        self.merger = PatchMerger(cfg, rngs=rngs)

    def _rotary(self, grid_thw, window_index):
        cfg = self.cfg
        # inv_freq is computed LOCALLY (host), NOT stored as self.inv_freq: as a non-Param numpy
        # attribute it leaked into the train state as an abstract ShapeDtypeStruct(20,) on the
        # full-ckpt restore path (head_dim//2=40 -> arange(0,40,2)=20 freqs), breaking shard_args.
        d = cfg.head_dim // 2
        inv_freq = 1.0 / (cfg.rope_theta ** (np.arange(0, d, 2, dtype=np.float32) / d))
        pos = rot_pos_ids(grid_thw, cfg.spatial_merge_size)          # [N,2]
        max_grid = int(grid_thw[:, 1:].max())
        seq = np.arange(max_grid, dtype=np.float32)
        full = np.outer(seq, inv_freq)                               # [max_grid, d/2]
        rpe = full[pos].reshape(pos.shape[0], -1)                    # [N, head_dim//2]
        # reorder by window_index (merged-unit granularity)
        u = cfg.spatial_merge_unit
        rpe = rpe.reshape(rpe.shape[0] // u, u, -1)[window_index].reshape(-1, rpe.shape[-1])
        emb = np.concatenate([rpe, rpe], -1)                         # [N, head_dim]
        return jnp.asarray(np.cos(emb)), jnp.asarray(np.sin(emb))

    def precompute_structural(self, grid_thw) -> dict:
        """Host-side geometry for the ViT forward, derived ONLY from ``grid_thw``
        (window partitioning / 2D-RoPE cos-sin / per-segment attention masks / output
        reorder). For a fixed image resolution these are constant across all samples, so
        they can be computed OFF-graph (in the data iterator) and fed into the jitted,
        differentiable :meth:`body`. No pixel data, no params — pure geometry.

        This is the EXACT structural half of the original ``__call__`` (same numpy ports),
        split out so the neural body can be jitted + differentiated (trainable ViT).
        Returns a dict of jnp arrays.
        """
        cfg = self.cfg
        grid_thw = np.asarray(grid_thw)
        N = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum())

        window_index, cu_win = get_window_index(grid_thw, cfg)
        cu_full = cu_seqlens_full(grid_thw)
        cos, sin = self._rotary(grid_thw, window_index)

        # segment masks: full = all patches one segment; window = per cu_win segment.
        seg_full = jnp.asarray(_seg_ids_from_cu(cu_full, N))
        mask_full = (seg_full[:, None] == seg_full[None, :])
        seg_win = jnp.asarray(_seg_ids_from_cu(cu_win, N))
        mask_win = (seg_win[:, None] == seg_win[None, :])

        return {
            "window_index": jnp.asarray(window_index),
            "cos": cos, "sin": sin,
            "mask_full": mask_full, "mask_win": mask_win,
            "rev": jnp.asarray(np.argsort(window_index)),
        }

    def body(self, pixel_values: Array, structural: dict) -> Array:
        """Pure-jax, differentiable ViT body: consumes ``pixel_values`` + the precomputed
        ``structural`` dict (from :meth:`precompute_structural`). Jittable end-to-end and
        the gradient flows to every ViT param (patch_embed / blocks / merger) — this is
        what makes the ViT *trainable* in-graph. Numerically identical to the original
        ``__call__`` (same ops, just no host-numpy inside).
        """
        cfg = self.cfg
        N = int(pixel_values.shape[0])
        u = cfg.spatial_merge_unit
        window_index = structural["window_index"]
        cos, sin = structural["cos"], structural["sin"]
        mask_full, mask_win = structural["mask_full"], structural["mask_win"]
        rev = structural["rev"]

        x = self.patch_embed(pixel_values)                          # [N, hidden]
        # reorder hidden by window_index (merged-unit granularity)
        x = x.reshape(N // u, u, -1)[window_index].reshape(N, -1)

        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, mask_full if i in cfg.fullatt_block_indexes else mask_win)

        x = self.merger(x)                                          # [N//u, out_hidden]
        return x[rev]                                               # reverse window reorder

    def __call__(self, pixel_values: Array, grid_thw: np.ndarray):
        # Backward-compatible split: structural precompute (host) + differentiable body.
        # Existing eval/iterator callers are unaffected (same inputs, same output, same
        # numerics). The trainable-ViT train path calls precompute_structural() in the
        # data iterator and body() inside the jitted graph.
        return self.body(pixel_values, self.precompute_structural(grid_thw))
