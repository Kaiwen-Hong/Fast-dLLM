"""5090 verification that the VISION pipeline conveys image CONTENT (isolated from scale).
Synthetic task: image = solid color (1 of 4 classes)+noise; FIXED question; answer = the color.
Text alone can't answer -> only the image can. If the tiny model learns it (acc>>25% and
correct-image >> wrong-image), the vision encoder demonstrably GROUNDS on the 5090.
Efficient: all jitted fns take single-example arrays (no python-int indexing inside jit)."""
import os
os.environ.setdefault("XLA_FLAGS","--xla_gpu_autotune_level=0"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
import sys; sys.path.insert(0,"/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma")
import numpy as np, jax, jax.numpy as jnp, optax
from kauldron import ktyping
from gemma import gm
from gemma.gm.nn.gemma4 import _config,_modules
from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput
from gemma.gm.nn.gemma4.vision import _encoder as ve,_preprocessing as pp
from gemma.diffusion import _models as dm
from hackable_diffusion import hd
from hackable_diffusion.lib.training import discrete_loss
NC=ktyping.config.Config(typechecking_enabled=False)
VOCAB,N_SOFT,Q_LEN=262144,16,8
K=4; COLORS=[(230,30,30),(30,200,30),(40,80,230),(230,210,30)]; NAMES=["red","green","blue","yellow"]
N_TRAIN,N_EVAL,STEPS=200,40,300
tok=gm.text.Gemma4Tokenizer(); st=gm.text.Gemma4Tokenizer.special_tokens
ANS=[tok.encode(" "+n,add_bos=False)[0] for n in NAMES]; print("class tokens:",ANS)
Q=tok.encode("What is the color?",add_bos=False)[:Q_LEN]; Q=Q+[0]*(Q_LEN-len(Q))
PROMPT=np.asarray([int(st.BOS),int(st.START_OF_IMAGE)]+[ve.TOKEN_PLACEHOLDER]*N_SOFT+[int(st.END_OF_IMAGE)]+Q,np.int32)
PROMPT_LEN=len(PROMPT); SEQ_LEN=PROMPT_LEN+1
def make(rng,cls):
    base=np.array(COLORS[cls],np.float32)
    img=np.clip(base[None,None,:]+np.asarray(jax.random.normal(rng,(64,64,3)))*18.0,0,255).astype(np.uint8)
    ptc,pos,_=pp.preprocess_and_patchify([img],patch_size=16,max_soft_tokens=16,pooling_kernel_size=1)
    return np.reshape(ptc,(ptc.shape[1],ptc.shape[2])),np.reshape(pos,(pos.shape[1],pos.shape[2]))
def build(n,seed):
    r=jax.random.PRNGKey(seed); P,X,Y,C=[],[],[],[]
    for i in range(n):
        r,s=jax.random.split(r); c=i%K; p,x=make(s,c); P.append(p); X.append(x); Y.append(ANS[c]); C.append(c)
    return (jnp.asarray(np.stack(P)),jnp.asarray(np.stack(X)),jnp.asarray(np.asarray(Y,np.int32)),np.asarray(C))
trP,trX,trY,trC=build(N_TRAIN,1); evP,evX,evY,evC=build(N_EVAL,2)
print(f"built {N_TRAIN} train / {N_EVAL} eval, {K} classes")
venc=ve.VisionEncoder(d_model=64,num_layers=2,num_heads=2,ffw_hidden=128,patch_size=16,output_length=16,pooling_kernel_size=1)
attn=_config.make_attention_layers_types((_modules.AttentionType.LOCAL_SLIDING,),num_layers=2)
cfg=_config.TransformerConfig(num_embed=VOCAB,embed_dim=128,hidden_dim=256,num_heads=4,head_dim=32,num_kv_heads=1,final_logit_softcap=30.0,use_post_attn_norm=True,use_post_ffw_norm=True,attention_types=attn,sliding_window_size=4096,qk_norm_with_scale=True,global_rope_proportion=0.25,local_rope_proportion=1.0,per_layer_input_dim=0,enable_moe=False,vision_encoder=venc,audio_encoder=None,use_bidirectional_attention=None)
model=dm.DiffusionGemma_26B_A4B(config=cfg,dtype=jnp.float32,text_only=False)
corr=hd.corruption.CategoricalProcess.uniform_process(num_categories=VOCAB,schedule=hd.corruption.RFSchedule())
tsamp=hd.training.time_sampling.UniformTimeSampler(span=hd.jax_helpers.SafeSpan(safety_epsilon=1e-4))
positions=jnp.arange(SEQ_LEN)[None]; amk=jnp.ones((1,SEQ_LEN,SEQ_LEN),bool); sc0=jnp.zeros((1,SEQ_LEN,128),jnp.float32); promptB=jnp.asarray(PROMPT)[None]; ANS_ARR=jnp.asarray(ANS)
def fwd(params,xt_tok,patches_i,posxy_i):
    pvi=PreprocessedVisionInput(patches=patches_i[None],positions_xy=posxy_i[None],soft_token_counts=(N_SOFT,))
    toks=jnp.concatenate([promptB,xt_tok[None,None]],axis=1)
    with NC:
        o=model.apply(params,toks,sc_embeddings=sc0,images=pvi,positions=positions,attention_mask=amk,sliding_attention_mask=amk,method=dm.DiffusionGemma_26B_A4B.call_with_self_conditioning)
    return o.logits[0,PROMPT_LEN]  # [V] logits at the single answer position
def loss_one(params,ans,patches_i,posxy_i,rng):
    x0=jnp.asarray(ans).reshape(1,1,1); t=tsamp(rng,x0); xt,tgt=corr.corrupt(rng,x0,t)
    lg=fwd(params,xt[0,0,0],patches_i,posxy_i)[None,None]  # [1,1,V]
    tgt["target_mask"]=jnp.ones((1,1,1),bool)
    return jnp.asarray(discrete_loss.compute_discrete_diffusion_loss(preds={"logits":lg},targets=tgt,time=t,use_mask=True,mask_key="target_mask")).mean()
@jax.jit
def step_fn(params,ost,ans,patches_i,posxy_i,r):
    l,g=jax.value_and_grad(loss_one)(params,ans,patches_i,posxy_i,r); u,ost=opt.update(g,ost); return optax.apply_updates(params,u),ost,l
@jax.jit
def predict_class(params,patches_i,posxy_i,rng):
    # fully-corrupt the answer token (t=1) so only image+question inform it; argmax among K class tokens
    x0=jnp.asarray(ANS[0]).reshape(1,1,1); t=jnp.ones_like(tsamp(rng,x0)); xt,_=corr.corrupt(rng,x0,t)
    lg=fwd(params,xt[0,0,0],patches_i,posxy_i); return jnp.argmax(lg[ANS_ARR])
loss_j=jax.jit(loss_one)
rng=jax.random.PRNGKey(0)
x0=jnp.asarray(trY[0]).reshape(1,1,1); t0=tsamp(rng,x0); xt0,_=corr.corrupt(rng,x0,t0)
with NC:
    pvi0=PreprocessedVisionInput(patches=trP[0][None],positions_xy=trX[0][None],soft_token_counts=(N_SOFT,))
    params=model.init(rng,jnp.concatenate([promptB,xt0[0,0,0].reshape(1,1)],axis=1),sc_embeddings=sc0,images=pvi0,positions=positions,attention_mask=amk,sliding_attention_mask=amk,method=dm.DiffusionGemma_26B_A4B.call_with_self_conditioning)
opt=optax.adam(3e-3); ost=opt.init(params)
def evaluate(params):
    rc=jax.random.PRNGKey(7); aC=aW=0; lC=lW=0.0
    for i in range(N_EVAL):
        r=jax.random.fold_in(rc,i); j=(i+1)%N_EVAL
        while evC[j]==evC[i]: j=(j+1)%N_EVAL   # wrong image = a DIFFERENT color
        aC+=int(int(predict_class(params,evP[i],evX[i],r))==evC[i])
        aW+=int(int(predict_class(params,evP[j],evX[j],r))==evC[i])
        lC+=float(loss_j(params,evY[i],evP[i],evX[i],r)); lW+=float(loss_j(params,evY[i],evP[j],evX[j],r))
    return aC/N_EVAL,aW/N_EVAL,lC/N_EVAL,lW/N_EVAL
print(f"training {STEPS} steps (chance acc=25%) ...")
for step in range(1,STEPS+1):
    i=(step-1)%N_TRAIN; rng,s=jax.random.split(rng)
    params,ost,loss=step_fn(params,ost,trY[i],trP[i],trX[i],s)
    if step in (50,100,150,200,300):
        aC,aW,lC,lW=evaluate(params)
        print(f"  step {step:3d}: acc(correct)={aC:.1%} acc(wrong)={aW:.1%} | loss correct={lC:.3f} wrong={lW:.3f} gap={lW-lC:+.3f}")
aC,aW,lC,lW=evaluate(params)
print(f"\nFINAL: correct-image acc={aC:.1%} (chance 25%) | wrong-image acc={aW:.1%} | loss gap(wrong-correct)={lW-lC:+.3f}")
print("VISION GROUNDS ON 5090:", (aC>0.6 and aC>aW+0.25 and (lW-lC)>0.2))

# ---- weight-ablation: are the TRAINED vision weights specifically what enable grounding? ----
def scramble(params, which):
    flat, tree = jax.tree_util.tree_flatten_with_path(params)
    ks = jax.random.split(jax.random.PRNGKey(4242), len(flat))
    new = [(jax.random.normal(ks[i], v.shape, v.dtype)*0.02 if which in "/".join(str(x) for x in p).lower() else v)
           for i,(p,v) in enumerate(flat)]
    return jax.tree_util.tree_unflatten(tree, new)
aCv,aWv,lCv,lWv = evaluate(scramble(params, "vision"))
aCt,aWt,_,_     = evaluate(scramble(params, "layer_"))  # scramble text transformer blocks (control)
print(f"ABLATE: scramble TRAINED VISION weights -> acc(correct)={aCv:.1%} gap={lWv-lCv:+.3f}  (grounding {aC:.0%}->{aCv:.0%})")
print(f"ABLATE: scramble text-block weights       -> acc(correct)={aCt:.1%}                    (control)")
print("TRAINED-VISION-WEIGHTS-ARE-ESSENTIAL:", aCv < aC-0.25 and aCv < 0.45)
