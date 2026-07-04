"""Phase 2 — VQA (ChartQA) diffusion training on DiffusionGemma with IMAGE input.

Self-contained: tiny random-init DiffusionGemma + tiny vision encoder, real ChartQA
(image+question->prompt with vision soft-tokens, answer->diffusion canvas), hackable_diffusion
uniform categorical corruption + discrete diffusion loss. Trains 100 steps, saves params at
steps 10/25/50/100, evaluates the diffusion loss on a held-out set at each checkpoint.

Requires the two upstream patches documented in gpu_smoke/PHASE2_VQA.md:
 - vision/_images.py factorized_posemb: jnp.nan -> 0 (masked anyway)
 - vision/_token_utils.remove_mm_logits: identity (Gemma4 path hardcodes Gemma3 tokens -> gather NaN)
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
print("jax", jax.__version__, jax.default_backend(), jax.devices())
VOCAB = 262_144
N_SOFT = 16          # square image -> 16 vision soft tokens
Q_LEN = 24           # padded question length
C_LEN = 12           # answer canvas length
N_TRAIN, N_EVAL = 100, 20
CKPT_STEPS = {10, 25, 50, 100}
OUT = "/home/kaiwen/data/overnight_dgemma/xp_chartqa"
os.makedirs(OUT, exist_ok=True)

tok = gm.text.Gemma4Tokenizer(); st = gm.text.Gemma4Tokenizer.special_tokens
PROMPT_HEAD = [int(st.BOS), int(st.START_OF_IMAGE)] + [ve.TOKEN_PLACEHOLDER]*N_SOFT + [int(st.END_OF_IMAGE)]
PROMPT_LEN = len(PROMPT_HEAD) + Q_LEN
SEQ_LEN = PROMPT_LEN + C_LEN
PAD = 0

def prep(ex):
    img = Image.open(io.BytesIO(ex["image"])).convert("RGB").resize((64, 64))
    ptc, pos, sc = pp.preprocess_and_patchify([np.asarray(img)], patch_size=16,
                                              max_soft_tokens=16, pooling_kernel_size=1)
    patches = np.reshape(ptc, (ptc.shape[0]*ptc.shape[1], ptc.shape[2]))     # [16,768]
    posxy = np.reshape(pos, (pos.shape[0]*pos.shape[1], pos.shape[2]))       # [16,2]
    q = tok.encode(ex["query"], add_bos=False)[:Q_LEN]
    q = q + [PAD]*(Q_LEN-len(q))
    a = tok.encode(" "+str(ex["label"]), add_bos=False)[:C_LEN]
    amask = [1]*len(a) + [0]*(C_LEN-len(a))
    a = a + [PAD]*(C_LEN-len(a))
    prompt = PROMPT_HEAD + q
    return dict(patches=np.asarray(patches, np.float32), posxy=np.asarray(posxy, np.int32),
                prompt=np.asarray(prompt, np.int32), answer=np.asarray(a, np.int32),
                amask=np.asarray(amask, np.float32))

print("loading ChartQA ...")
ds = datasets.load_dataset("ahmed-masry/ChartQA", split="train", streaming=True)
rows = []
for ex in ds:
    rows.append(prep(ex))
    if len(rows) >= N_TRAIN + N_EVAL: break
def stack(rs, k): return jnp.asarray(np.stack([r[k] for r in rs]))
tr = {k: stack(rows[:N_TRAIN], k) for k in rows[0]}
ev = {k: stack(rows[N_TRAIN:], k) for k in rows[0]}
print(f"prepared {len(rows)} examples | seq_len={SEQ_LEN} (prompt {PROMPT_LEN} + canvas {C_LEN})")

# ---- tiny model ----
venc = ve.VisionEncoder(d_model=64, num_layers=2, num_heads=2, ffw_hidden=128,
                        patch_size=16, output_length=16, pooling_kernel_size=1)
attn = _config.make_attention_layers_types((_modules.AttentionType.LOCAL_SLIDING,), num_layers=2)
cfg = _config.TransformerConfig(
    num_embed=VOCAB, embed_dim=128, hidden_dim=256, num_heads=4, head_dim=32, num_kv_heads=1,
    final_logit_softcap=30.0, use_post_attn_norm=True, use_post_ffw_norm=True,
    attention_types=attn, sliding_window_size=4096, qk_norm_with_scale=True,
    global_rope_proportion=0.25, local_rope_proportion=1.0, per_layer_input_dim=0,
    enable_moe=False, vision_encoder=venc, audio_encoder=None, use_bidirectional_attention=None)
model = dm.DiffusionGemma_26B_A4B(config=cfg, dtype=jnp.float32, text_only=False)
corruption = hd.corruption.CategoricalProcess.uniform_process(num_categories=VOCAB, schedule=hd.corruption.RFSchedule())
time_sampler = hd.training.time_sampling.UniformTimeSampler(span=hd.jax_helpers.SafeSpan(safety_epsilon=1e-4))

def pvi_of(b, i):
    return PreprocessedVisionInput(patches=b["patches"][i][None], positions_xy=b["posxy"][i][None],
                                   soft_token_counts=(N_SOFT,))

positions = jnp.arange(SEQ_LEN)[None]
attn_mask = jnp.ones((1, SEQ_LEN, SEQ_LEN), bool)
sc0 = jnp.zeros((1, SEQ_LEN, cfg.embed_dim), jnp.float32)

def forward(params, prompt, xt_canvas, pvi):
    tokens = jnp.concatenate([prompt[None], xt_canvas[None, :, 0]], axis=1)  # [1, SEQ_LEN]
    with NC:
        out = model.apply(params, tokens, sc_embeddings=sc0, images=pvi, positions=positions,
                          attention_mask=attn_mask, sliding_attention_mask=attn_mask,
                          method=dm.DiffusionGemma_26B_A4B.call_with_self_conditioning)
    return out.logits[:, PROMPT_LEN:, :]   # logits over the canvas region [1, C_LEN, V]

def loss_one(params, ex, pvi, rng):
    x0 = ex["answer"][:, None]                       # [C_LEN,1]
    t = time_sampler(rng, x0[None])
    xt, tgt = corruption.corrupt(rng, x0[None], t)   # [1,C_LEN,1]
    logits = forward(params, ex["prompt"], xt[0], pvi)
    tgt["target_mask"] = (ex["amask"] > 0)[None, :, None]
    dl = discrete_loss.compute_discrete_diffusion_loss(
        preds={"logits": logits}, targets=tgt, time=t, use_mask=True, mask_key="target_mask")
    return jnp.asarray(dl).mean()

# init on example 0
rng = jax.random.PRNGKey(0)
ex0 = {k: tr[k][0] for k in tr}
x0 = ex0["answer"][:, None]; t0 = time_sampler(rng, x0[None]); xt0, _ = corruption.corrupt(rng, x0[None], t0)
with NC:
    params = model.init(rng, jnp.concatenate([ex0["prompt"][None], xt0[0][None,:,0]], axis=1),
                        sc_embeddings=sc0, images=pvi_of(tr, 0), positions=positions,
                        attention_mask=attn_mask, sliding_attention_mask=attn_mask,
                        method=dm.DiffusionGemma_26B_A4B.call_with_self_conditioning)
nparams = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"VQA model built: {nparams:,} params (tiny text + tiny vision)")

opt = optax.adam(3e-3); opt_state = opt.init(params)

@jax.jit
def train_step(params, opt_state, ex, pvi, rng):
    loss, grads = jax.value_and_grad(loss_one)(params, ex, pvi, rng)
    updates, opt_state = opt.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss

@jax.jit
def eval_loss(params, ex, pvi, rng):
    return loss_one(params, ex, pvi, rng)

def run_eval(params):
    rng_e = jax.random.PRNGKey(123)
    ls = []
    for i in range(N_EVAL):
        exi = {k: ev[k][i] for k in ev}
        ls.append(float(eval_loss(params, exi, pvi_of(ev, i), jax.random.fold_in(rng_e, i))))
    return float(np.mean(ls))

ckpt_eval = {}
print("training 100 steps (batch=1, cycling ChartQA) ...")
for step in range(1, 101):
    i = (step-1) % N_TRAIN
    exi = {k: tr[k][i] for k in tr}
    rng, sub = jax.random.split(rng)
    params, opt_state, loss = train_step(params, opt_state, exi, pvi_of(tr, i), sub)
    if step % 10 == 0 or step in CKPT_STEPS:
        print(f"  step {step:3d}: train_loss={float(loss):.4f}")
    if step in CKPT_STEPS:
        with open(f"{OUT}/ckpt_{step}.pkl", "wb") as f:
            pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
        el = run_eval(params); ckpt_eval[step] = el
        print(f"  >>> checkpoint {step}: eval_diffusion_loss={el:.4f}")

print("\n=== ChartQA VQA eval loss per checkpoint (metric = held-out diffusion loss) ===")
for s in sorted(ckpt_eval):
    print(f"  step {s:3d} ({s}%): eval_loss = {ckpt_eval[s]:.4f}")
vals = [ckpt_eval[s] for s in sorted(ckpt_eval)]
print("improving (monotone down):", all(vals[i] >= vals[i+1] for i in range(len(vals)-1)),
      "| overall down:", vals[0] > vals[-1])
print("PASS: VQA (ChartQA, image input) diffusion training ran 100 steps + evaluated 10/25/50/100% on GPU")
