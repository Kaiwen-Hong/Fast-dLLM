"""HF safetensors -> Flax NNX weight loader for the Fast-dDrive TEXT decoder.

Adapted from jax-mdlm-handoff/code/a2d.py. The Fast-dDrive checkpoint stores flat
Qwen2.5 keys: model.layers.N.self_attn.{q,k,v}_proj.{weight,bias},
self_attn.o_proj.weight, mlp.{gate,up,down}_proj.weight, input/post_attention_layernorm,
model.embed_tokens.weight, model.norm.weight, lm_head.weight (tied).

PyTorch Linear (out,in) -> JAX kernel (in,out): transpose weights, not biases/norms.
No mask-token mean-init needed (|<MASK>|=151665, <|NULL|>=151666 already trained).
"""
from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors import safe_open

from ..models.qwen2_5_text import Qwen25TextConfig, Qwen25TextModel


def _load_all_tensors(snapshot_dir: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for fname in sorted(os.listdir(snapshot_dir)):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(snapshot_dir, fname), framework="numpy") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


def _name_map_text(i: int) -> dict[str, tuple[str, bool]]:
    m = {
        f"model.layers.{i}.input_layernorm.weight": (f"layers/{i}/input_layernorm/weight", False),
        f"model.layers.{i}.post_attention_layernorm.weight": (f"layers/{i}/post_attention_layernorm/weight", False),
        f"model.layers.{i}.self_attn.q_proj.weight": (f"layers/{i}/self_attn/q_proj/kernel", True),
        f"model.layers.{i}.self_attn.k_proj.weight": (f"layers/{i}/self_attn/k_proj/kernel", True),
        f"model.layers.{i}.self_attn.v_proj.weight": (f"layers/{i}/self_attn/v_proj/kernel", True),
        f"model.layers.{i}.self_attn.o_proj.weight": (f"layers/{i}/self_attn/o_proj/kernel", True),
        f"model.layers.{i}.self_attn.q_proj.bias": (f"layers/{i}/self_attn/q_proj/bias", False),
        f"model.layers.{i}.self_attn.k_proj.bias": (f"layers/{i}/self_attn/k_proj/bias", False),
        f"model.layers.{i}.self_attn.v_proj.bias": (f"layers/{i}/self_attn/v_proj/bias", False),
        f"model.layers.{i}.mlp.gate_proj.weight": (f"layers/{i}/mlp/gate_proj/kernel", True),
        f"model.layers.{i}.mlp.up_proj.weight": (f"layers/{i}/mlp/up_proj/kernel", True),
        f"model.layers.{i}.mlp.down_proj.weight": (f"layers/{i}/mlp/down_proj/kernel", True),
    }
    return m


def _set(model, path: str, value: np.ndarray):
    obj = model
    parts = path.split("/")
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    leaf = getattr(obj, parts[-1])
    leaf.value = jnp.asarray(value).astype(leaf.value.dtype)


def load_fast_ddrive_text(snapshot_dir: str, cfg: Qwen25TextConfig | None = None,
                          *, dtype=jnp.float32, rngs: nnx.Rngs | None = None,
                          verbose: bool = True) -> tuple[Qwen25TextModel, dict]:
    cfg = cfg or Qwen25TextConfig.fast_ddrive(dtype=dtype)
    rngs = rngs or nnx.Rngs(0)
    tensors = _load_all_tensors(snapshot_dir)
    if verbose:
        print(f"[convert] {len(tensors)} tensors; building Qwen25TextModel "
              f"({cfg.n_layers}L/{cfg.d_model}d, vocab {cfg.vocab_size}, dtype {dtype.__name__})")
    model = Qwen25TextModel(cfg, rngs=rngs)

    n_loaded, n_missing = 0, []
    for i in range(cfg.n_layers):
        for hf, (path, tr) in _name_map_text(i).items():
            if hf not in tensors:
                n_missing.append(hf); continue
            arr = tensors[hf]
            if tr:
                arr = arr.T
            _set(model, path, arr)
            n_loaded += 1
    _set(model, "embed_tokens/embedding", tensors["model.embed_tokens.weight"]); n_loaded += 1
    _set(model, "norm/weight", tensors["model.norm.weight"]); n_loaded += 1

    # Tie check: confirm stored lm_head.weight matches embed (so embed.attend == PyTorch lm_head).
    tie_ok = None
    if "lm_head.weight" in tensors:
        e = tensors["model.embed_tokens.weight"].astype(np.float32)
        h = tensors["lm_head.weight"].astype(np.float32)
        if e.shape == h.shape:
            tie_ok = float(np.abs(e - h).max())
            if verbose:
                print(f"[convert] tie check: max|embed - lm_head| = {tie_ok:.3e} "
                      f"({'TIED' if tie_ok < 1e-5 else 'UNTIED — using stored lm_head'})")
        if tie_ok is not None and tie_ok >= 1e-5 and not cfg.tie_embeddings:
            _set(model, "lm_head/kernel", h.T); n_loaded += 1

    info = {"n_loaded": n_loaded, "n_missing": n_missing, "tie_max_abs": tie_ok,
            "vit_keys": sum(1 for k in tensors if k.startswith("visual.")),
            "total_keys": len(tensors)}
    if verbose:
        print(f"[convert] loaded {n_loaded} text tensors; {len(n_missing)} missing; "
              f"{info['vit_keys']} visual.* keys deferred (Phase 4)")
    return model, info


def load_fast_ddrive_vit(snapshot_dir: str, cfg=None, *, dtype=jnp.float32,
                         rngs: nnx.Rngs | None = None, verbose: bool = True):
    """Load the Qwen2.5-VL vision tower weights into the JAX VisionTransformer."""
    from ..models.vision_qwen25vl import VisionConfig, VisionTransformer
    cfg = cfg or VisionConfig(dtype=dtype)
    rngs = rngs or nnx.Rngs(0)
    tensors = _load_all_tensors(snapshot_dir)
    vit = VisionTransformer(cfg, rngs=rngs)
    n = 0

    def setv(path, arr):
        nonlocal n
        _set(vit, path, arr); n += 1

    # patch_embed: Conv3d [out,in,T,H,W] -> Lin kernel [in_flat, out]
    pe = tensors["visual.patch_embed.proj.weight"]
    setv("patch_embed/kernel", pe.reshape(pe.shape[0], -1).T)
    for i in range(cfg.depth):
        p = f"visual.blocks.{i}."
        q = f"blocks/{i}/"
        setv(q + "norm1/weight", tensors[p + "norm1.weight"])
        setv(q + "norm2/weight", tensors[p + "norm2.weight"])
        setv(q + "attn/qkv/kernel", tensors[p + "attn.qkv.weight"].T)
        setv(q + "attn/qkv/bias", tensors[p + "attn.qkv.bias"])
        setv(q + "attn/proj/kernel", tensors[p + "attn.proj.weight"].T)
        setv(q + "attn/proj/bias", tensors[p + "attn.proj.bias"])
        for m in ("gate_proj", "up_proj", "down_proj"):
            setv(q + f"mlp/{m}/kernel", tensors[p + f"mlp.{m}.weight"].T)
            setv(q + f"mlp/{m}/bias", tensors[p + f"mlp.{m}.bias"])
    setv("merger/ln_q/weight", tensors["visual.merger.ln_q.weight"])
    setv("merger/fc1/kernel", tensors["visual.merger.mlp.0.weight"].T)
    setv("merger/fc1/bias", tensors["visual.merger.mlp.0.bias"])
    setv("merger/fc2/kernel", tensors["visual.merger.mlp.2.weight"].T)
    setv("merger/fc2/bias", tensors["visual.merger.mlp.2.bias"])
    if verbose:
        print(f"[convert-vit] loaded {n} vision tensors (depth {cfg.depth})")
    return vit, {"n_loaded": n}
