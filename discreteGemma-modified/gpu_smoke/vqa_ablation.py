"""Phase 2 verification: does the VISION encoder actually contribute, or is the
loss drop just text/answer-prior overfitting?

Controlled ablations on the trained ckpt_100:
  T1 counterfactual image  : held-out diffusion loss with (correct | shuffled/wrong | zero) image,
                             SAME rng per example so the ONLY difference is the image content.
  T1b logit sensitivity    : ||answer-logits(correct) - answer-logits(zero)|| (0 => image ignored).
  T2 gradient flow         : ||grad|| into the vision-encoder params vs the text params.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("HF_HOME", "/home/kaiwen/data/huggingface")
import sys, io, pickle
sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma")
import numpy as np, jax, jax.numpy as jnp, optax
from PIL import Image
import datasets
from kauldron import ktyping
from gemma import gm
from gemma.gm.nn.gemma4 import _config, _modules
from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput
from gemma.gm.nn.gemma4.vision import _encoder as ve, _preprocessing as pp
from gemma.diffusion import _models as dm
from hackable_diffusion import hd
from hackable_diffusion.lib.training import discrete_loss

NC = ktyping.config.Config(typechecking_enabled=False)
VOCAB, N_SOFT, Q_LEN, C_LEN = 262_144, 16, 24, 12
N_TRAIN, N_EVAL = 100, 20
tok = gm.text.Gemma4Tokenizer(); st = gm.text.Gemma4Tokenizer.special_tokens
PROMPT_HEAD = [int(st.BOS), int(st.START_OF_IMAGE)] + [ve.TOKEN_PLACEHOLDER]*N_SOFT + [int(st.END_OF_IMAGE)]
PROMPT_LEN = len(PROMPT_HEAD) + Q_LEN
SEQ_LEN = PROMPT_LEN + C_LEN
PAD = 0

def prep(ex):
    img = Image.open(io.BytesIO(ex["image"])).convert("RGB").resize((64, 64))
    ptc, pos, sc = pp.preprocess_and_patchify([np.asarray(img)], patch_size=16, max_soft_tokens=16, pooling_kernel_size=1)
    patches = np.reshape(ptc, (ptc.shape[0]*ptc.shape[1], ptc.shape[2]))
    posxy = np.reshape(pos, (pos.shape[0]*pos.shape[1], pos.shape[2]))
    q = tok.encode(ex["query"], add_bos=False)[:Q_LEN]; q += [PAD]*(Q_LEN-len(q))
    a = tok.encode(" "+str(ex["label"]), add_bos=False)[:C_LEN]
    amask = [1]*len(a) + [0]*(C_LEN-len(a)); a += [PAD]*(C_LEN-len(a))
    return dict(patches=np.asarray(patches, np.float32), posxy=np.asarray(posxy, np.int32),
                prompt=np.asarray(PROMPT_HEAD+q, np.int32), answer=np.asarray(a, np.int32),
                amask=np.asarray(amask, np.float32))

ds = datasets.load_dataset("ahmed-masry/ChartQA", split="train", streaming=True)
rows = []
for ex in ds:
    rows.append(prep(ex))
    if len(rows) >= N_TRAIN+N_EVAL: break
def stack(rs, k): return jnp.asarray(np.stack([r[k] for r in rs]))
ev = {k: stack(rows[N_TRAIN:], k) for k in rows[0]}
print("jax", jax.__version__, jax.default_backend(), jax.devices(), "| eval N =", N_EVAL)

venc = ve.VisionEncoder(d_model=64, num_layers=2, num_heads=2, ffw_hidden=128, patch_size=16, output_length=16, pooling_kernel_size=1)
attn = _config.make_attention_layers_types((_modules.AttentionType.LOCAL_SLIDING,), num_layers=2)
cfg = _config.TransformerConfig(num_embed=VOCAB, embed_dim=128, hidden_dim=256, num_heads=4, head_dim=32, num_kv_heads=1,
    final_logit_softcap=30.0, use_post_attn_norm=True, use_post_ffw_norm=True, attention_types=attn, sliding_window_size=4096,
    qk_norm_with_scale=True, global_rope_proportion=0.25, local_rope_proportion=1.0, per_layer_input_dim=0, enable_moe=False,
    vision_encoder=venc, audio_encoder=None, use_bidirectional_attention=None)
model = dm.DiffusionGemma_26B_A4B(config=cfg, dtype=jnp.float32, text_only=False)
corruption = hd.corruption.CategoricalProcess.uniform_process(num_categories=VOCAB, schedule=hd.corruption.RFSchedule())
time_sampler = hd.training.time_sampling.UniformTimeSampler(span=hd.jax_helpers.SafeSpan(safety_epsilon=1e-4))
positions = jnp.arange(SEQ_LEN)[None]; attn_mask = jnp.ones((1, SEQ_LEN, SEQ_LEN), bool); sc0 = jnp.zeros((1, SEQ_LEN, 128), jnp.float32)

with open("/home/kaiwen/data/overnight_dgemma/xp_chartqa/ckpt_100.pkl", "rb") as f:
    params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
print("loaded ckpt_100")

def pvi_from(patches, posxy):
    return PreprocessedVisionInput(patches=patches[None], positions_xy=posxy[None], soft_token_counts=(N_SOFT,))

def fwd_logits(params, prompt, xt_canvas, pvi):
    tokens = jnp.concatenate([prompt[None], xt_canvas[None, :, 0]], axis=1)
    with NC:
        out = model.apply(params, tokens, sc_embeddings=sc0, images=pvi, positions=positions,
                          attention_mask=attn_mask, sliding_attention_mask=attn_mask,
                          method=dm.DiffusionGemma_26B_A4B.call_with_self_conditioning)
    return out.logits[:, PROMPT_LEN:, :]

def loss_with(params, ex, patches, posxy, rng):
    x0 = ex["answer"][:, None]; t = time_sampler(rng, x0[None]); xt, tgt = corruption.corrupt(rng, x0[None], t)
    logits = fwd_logits(params, ex["prompt"], xt[0], pvi_from(patches, posxy))
    tgt["target_mask"] = (ex["amask"] > 0)[None, :, None]
    return jnp.asarray(discrete_loss.compute_discrete_diffusion_loss(preds={"logits": logits}, targets=tgt, time=t, use_mask=True, mask_key="target_mask")).mean()

loss_jit = jax.jit(loss_with)

# ---- T1: counterfactual image (same rng per example) ----
zero_patches = jnp.zeros_like(ev["patches"][0]); zero_pos = ev["posxy"][0]
Lc, Ls, Lz = [], [], []
rng0 = jax.random.PRNGKey(999)
for i in range(N_EVAL):
    exi = {k: ev[k][i] for k in ev}; r = jax.random.fold_in(rng0, i)
    j = (i + 7) % N_EVAL   # a DIFFERENT example's image (wrong chart)
    Lc.append(float(loss_jit(params, exi, ev["patches"][i], ev["posxy"][i], r)))
    Ls.append(float(loss_jit(params, exi, ev["patches"][j], ev["posxy"][j], r)))
    Lz.append(float(loss_jit(params, exi, zero_patches, zero_pos, r)))
import statistics as S
print("\n=== T1 counterfactual eval diffusion loss (same rng/example; only the IMAGE differs) ===")
print(f"  correct image : {S.mean(Lc):.4f}")
print(f"  wrong  image  : {S.mean(Ls):.4f}   (delta vs correct: {S.mean(Ls)-S.mean(Lc):+.4f})")
print(f"  zero   image  : {S.mean(Lz):.4f}   (delta vs correct: {S.mean(Lz)-S.mean(Lc):+.4f})")
per_ex_better = sum(1 for a,b in zip(Lc,Ls) if a < b)
print(f"  correct < wrong on {per_ex_better}/{N_EVAL} examples")

# ---- T1b: logit sensitivity to image content ----
diffs = []
for i in range(min(8, N_EVAL)):
    exi = {k: ev[k][i] for k in ev}; xt = exi["answer"][:, None]
    lc = fwd_logits(params, exi["prompt"], xt, pvi_from(ev["patches"][i], ev["posxy"][i]))
    lz = fwd_logits(params, exi["prompt"], xt, pvi_from(zero_patches, zero_pos))
    diffs.append(float(jnp.sqrt(((lc-lz)**2).mean())))
print(f"\n=== T1b answer-logit RMS diff (correct vs zero image): {S.mean(diffs):.4f} (0 => image ignored) ===")

# ---- T2: gradient flow into vision params ----
exi = {k: ev[k][0] for k in ev}
g = jax.grad(loss_with)(params, exi, ev["patches"][0], ev["posxy"][0], jax.random.PRNGKey(1))
def gnorm(tree, pred):
    leaves = [jnp.sum(v**2) for p, v in jax.tree_util.tree_flatten_with_path(tree)[0] if pred("/".join(str(x) for x in p))]
    return float(jnp.sqrt(sum(leaves))) if leaves else 0.0
vg = gnorm(g, lambda s: "vision" in s.lower()); tg = gnorm(g, lambda s: "vision" not in s.lower())
print(f"\n=== T2 gradient norm: vision-encoder params = {vg:.4e} | non-vision params = {tg:.4e} ===")
print("  vision grads nonzero:", vg > 0)
