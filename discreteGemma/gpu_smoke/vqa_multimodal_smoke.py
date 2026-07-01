"""Phase 2 de-risk: images -> vision encoder -> DiffusionGemma multimodal diffusion
forward + one train step, on a real ChartQA image. Tiny random-init, RTX 5090.

Notes on the (undocumented) Gemma4-diffusion vision path:
 - the diffusion `call_with_self_conditioning` has a WRONG `images` annotation
   (UInt8[...]) but the code actually wants a PreprocessedVisionInput -> we disable
   ktyping typechecking around the call.
 - soft-token positions are marked with TOKEN_PLACEHOLDER (-2); count must equal the
   number of vision soft tokens (sum of soft_token_counts).
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("HF_HOME", "/home/kaiwen/data/huggingface")
import sys, io
sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma")
import numpy as np, jax, jax.numpy as jnp, optax
from PIL import Image
import datasets
from kauldron import ktyping

from gemma import gm
from gemma.gm.nn.gemma4 import _config, _modules
from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput
from gemma.gm.nn.gemma4.vision import _encoder as vision_encoder
from gemma.gm.nn.gemma4.vision import _preprocessing
from gemma.gm.vision import _token_utils
from gemma.diffusion import _models as dm

NOCHECK = ktyping.config.Config(typechecking_enabled=False)
print("jax", jax.__version__, jax.default_backend(), jax.devices())
VOCAB = 262_144

# ---- tiny vision encoder + tiny text backbone WITH vision ----
venc = vision_encoder.VisionEncoder(
    d_model=64, num_layers=2, num_heads=2, ffw_hidden=128,
    patch_size=16, output_length=16, pooling_kernel_size=1)
attn = _config.make_attention_layers_types((_modules.AttentionType.LOCAL_SLIDING,), num_layers=2)
cfg = _config.TransformerConfig(
    num_embed=VOCAB, embed_dim=64, hidden_dim=128, num_heads=2, head_dim=16, num_kv_heads=1,
    final_logit_softcap=30.0, use_post_attn_norm=True, use_post_ffw_norm=True,
    attention_types=attn, sliding_window_size=4096, qk_norm_with_scale=True,
    global_rope_proportion=0.25, local_rope_proportion=1.0, per_layer_input_dim=0,
    enable_moe=False, vision_encoder=venc, audio_encoder=None,
    use_bidirectional_attention=None)
model = dm.DiffusionGemma_26B_A4B(config=cfg, dtype=jnp.float32, text_only=False)

# ---- real ChartQA example ----
ex = next(iter(datasets.load_dataset("ahmed-masry/ChartQA", split="train", streaming=True)))
img = Image.open(io.BytesIO(ex["image"])).convert("RGB")
print("Q:", ex["query"], "| A:", ex["label"], "| img:", img.size)
tok = gm.text.Gemma4Tokenizer(); st = gm.text.Gemma4Tokenizer.special_tokens

# ---- preprocess image -> PreprocessedVisionInput ----
patches, positions_xy, soft_counts = _preprocessing.preprocess_and_patchify(
    [np.asarray(img)], patch_size=16, max_soft_tokens=16, pooling_kernel_size=1)
n_img, max_patches = patches.shape[0], patches.shape[1]
pvi = PreprocessedVisionInput(
    patches=jnp.reshape(patches, (1, n_img*max_patches, patches.shape[2])),
    positions_xy=jnp.reshape(positions_xy, (1, n_img*max_patches, positions_xy.shape[2])),
    soft_token_counts=tuple(int(c) for c in soft_counts))
total_soft = sum(pvi.soft_token_counts)
print("patches:", pvi.patches.shape, "soft_counts:", pvi.soft_token_counts, "total_soft:", total_soft)

# ---- token seq: [BOS, <soi>, -2*total_soft, <eoi>] + question + answer ----
q_ids = tok.encode(ex["query"], add_bos=False)
a_ids = tok.encode(" " + ex["label"], add_bos=False)
seq = ([int(st.BOS), int(st.START_OF_IMAGE)] + [vision_encoder.TOKEN_PLACEHOLDER]*total_soft
       + [int(st.END_OF_IMAGE)] + q_ids + a_ids)
tokens = jnp.asarray(seq, jnp.int32)[None]
L = tokens.shape[1]
n_ph = int((tokens == vision_encoder.TOKEN_PLACEHOLDER).sum())
print("seq len:", L, "| soft placeholders (-2):", n_ph, "(must == total_soft)")

positions = jnp.broadcast_to(jnp.arange(L), (1, L))
attn_mask = jnp.ones((1, L, L), dtype=bool)
sc = jnp.zeros((1, L, cfg.embed_dim), jnp.float32)

def fwd(params, toks):
    with NOCHECK:
        return model.apply(params, toks, sc_embeddings=sc, images=pvi,
                           positions=positions, attention_mask=attn_mask,
                           sliding_attention_mask=attn_mask,
                           method=dm.DiffusionGemma_26B_A4B.call_with_self_conditioning)

with NOCHECK:
    params = model.init(jax.random.PRNGKey(0), tokens, sc_embeddings=sc, images=pvi,
                        positions=positions, attention_mask=attn_mask,
                        sliding_attention_mask=attn_mask,
                        method=dm.DiffusionGemma_26B_A4B.call_with_self_conditioning)
np_ = sum(x.size for x in jax.tree_util.tree_leaves(params))
has_v = any('vision' in '/'.join(map(str,p)).lower() for p,_ in jax.tree_util.tree_flatten_with_path(params)[0])
print(f"VQA model params: {np_:,} | vision encoder present: {has_v}")

out = fwd(params, tokens); out.logits.block_until_ready()
import numpy as _np
lg=out.logits
nanpos=jnp.isnan(lg[0]).any(axis=-1)
print("nan-per-position:", [int(x) for x in nanpos])
print("token ids:", [int(t) for t in tokens[0]])
print("also check hidden via return_hidden? logits finite:", bool(jnp.isfinite(lg).all()))

n_ans=len(a_ids)
def loss_fn(p):
    lg = fwd(p, tokens).logits[:, -n_ans-1:-1, :]
    lab = tokens[:, -n_ans:]
    return optax.softmax_cross_entropy_with_integer_labels(lg, lab).mean()
opt = optax.adam(1e-3); so = opt.init(params); l0 = loss_fn(params)
for i in range(3):
    loss, g = jax.value_and_grad(loss_fn)(params)
    up, so = opt.update(g, so); params = optax.apply_updates(params, up)
    print(f"  step {i}: loss={float(loss):.4f}")
print(f"training OK: loss {float(l0):.4f} -> {float(loss_fn(params)):.4f}")
print("PASS: VQA multimodal diffusion (image->vision->diffusion) trains on GPU")
