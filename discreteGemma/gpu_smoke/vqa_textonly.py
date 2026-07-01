"""Control: TEXT-ONLY baseline (no image) — same text config/data/loss as vqa_train,
prompt = BOS + question, answer->diffusion canvas. If this matches the VQA model's eval loss,
the image adds nothing beyond text/answer-prior."""
import os
os.environ.setdefault("XLA_FLAGS","--xla_gpu_autotune_level=0"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2"); os.environ.setdefault("HF_HOME","/home/kaiwen/data/huggingface")
import sys; sys.path.insert(0,"/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma")
import numpy as np, jax, jax.numpy as jnp, optax
import datasets
from gemma import gm
from gemma.gm.nn.gemma4 import _config,_modules
from gemma.diffusion import _models as dm
from hackable_diffusion import hd
from hackable_diffusion.lib.training import discrete_loss
VOCAB,Q_LEN,C_LEN,N_TRAIN,N_EVAL=262144,24,12,100,20
tok=gm.text.Gemma4Tokenizer(); st=gm.text.Gemma4Tokenizer.special_tokens
PROMPT_LEN=1+Q_LEN; SEQ_LEN=PROMPT_LEN+C_LEN; PAD=0
def prep(ex):
    q=tok.encode(ex["query"],add_bos=False)[:Q_LEN]; q=q+[PAD]*(Q_LEN-len(q))
    a=tok.encode(" "+str(ex["label"]),add_bos=False)[:C_LEN]; am=[1]*len(a)+[0]*(C_LEN-len(a)); a=a+[PAD]*(C_LEN-len(a))
    return dict(prompt=np.asarray([int(st.BOS)]+q,np.int32),answer=np.asarray(a,np.int32),amask=np.asarray(am,np.float32))
rows=[]
for ex in datasets.load_dataset("ahmed-masry/ChartQA",split="train",streaming=True):
    rows.append(prep(ex))
    if len(rows)>=N_TRAIN+N_EVAL: break
def stack(rs,k): return jnp.asarray(np.stack([r[k] for r in rs]))
tr={k:stack(rows[:N_TRAIN],k) for k in rows[0]}; ev={k:stack(rows[N_TRAIN:],k) for k in rows[0]}
attn=_config.make_attention_layers_types((_modules.AttentionType.LOCAL_SLIDING,),num_layers=2)
cfg=_config.TransformerConfig(num_embed=VOCAB,embed_dim=128,hidden_dim=256,num_heads=4,head_dim=32,num_kv_heads=1,final_logit_softcap=30.0,use_post_attn_norm=True,use_post_ffw_norm=True,attention_types=attn,sliding_window_size=4096,qk_norm_with_scale=True,global_rope_proportion=0.25,local_rope_proportion=1.0,per_layer_input_dim=0,enable_moe=False,vision_encoder=None,audio_encoder=None)
model=dm.DiffusionGemma_26B_A4B(config=cfg,dtype=jnp.float32)
corr=hd.corruption.CategoricalProcess.uniform_process(num_categories=VOCAB,schedule=hd.corruption.RFSchedule())
tsamp=hd.training.time_sampling.UniformTimeSampler(span=hd.jax_helpers.SafeSpan(safety_epsilon=1e-4))
pos=jnp.arange(SEQ_LEN)[None]; amk=jnp.ones((1,SEQ_LEN,SEQ_LEN),bool); sc0=jnp.zeros((1,SEQ_LEN,128),jnp.float32)
def fwd(p,prompt,xt):
    toks=jnp.concatenate([prompt[None],xt[None,:,0]],axis=1)
    o=model.apply(p,toks,sc_embeddings=sc0,positions=pos,attention_mask=amk,sliding_attention_mask=amk,method=dm.DiffusionGemma_26B_A4B.call_with_self_conditioning)
    return o.logits[:,PROMPT_LEN:,:]
def loss_one(p,ex,rng):
    x0=ex["answer"][:,None]; t=tsamp(rng,x0[None]); xt,tgt=corr.corrupt(rng,x0[None],t)
    lg=fwd(p,ex["prompt"],xt[0]); tgt["target_mask"]=(ex["amask"]>0)[None,:,None]
    return jnp.asarray(discrete_loss.compute_discrete_diffusion_loss(preds={"logits":lg},targets=tgt,time=t,use_mask=True,mask_key="target_mask")).mean()
rng=jax.random.PRNGKey(0); ex0={k:tr[k][0] for k in tr}
x0=ex0["answer"][:,None]; t0=tsamp(rng,x0[None]); xt0,_=corr.corrupt(rng,x0[None],t0)
params=model.init(rng,jnp.concatenate([ex0["prompt"][None],xt0[0][None,:,0]],axis=1),sc_embeddings=sc0,positions=pos,attention_mask=amk,sliding_attention_mask=amk,method=dm.DiffusionGemma_26B_A4B.call_with_self_conditioning)
opt=optax.adam(3e-3); ost=opt.init(params)
@jax.jit
def step_fn(p,s,ex,r):
    l,g=jax.value_and_grad(loss_one)(p,ex,r); u,s=opt.update(g,s); return optax.apply_updates(p,u),s,l
eval_fn=jax.jit(loss_one)
def run_eval(p):
    re=jax.random.PRNGKey(123); return float(np.mean([float(eval_fn(p,{k:ev[k][i] for k in ev},jax.random.fold_in(re,i))) for i in range(N_EVAL)]))
res={}
for step in range(1,101):
    i=(step-1)%N_TRAIN; exi={k:tr[k][i] for k in tr}; rng,sub=jax.random.split(rng)
    params,ost,loss=step_fn(params,ost,exi,sub)
    if step in (10,25,50,100): res[step]=run_eval(params)
print("=== TEXT-ONLY baseline eval diffusion loss (no image) ===")
for s in sorted(res): print(f"  step {s:3d}: eval_loss={res[s]:.4f}")
