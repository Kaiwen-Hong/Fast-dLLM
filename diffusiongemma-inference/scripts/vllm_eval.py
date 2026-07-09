import os, time, re
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER","1"); os.environ.setdefault("HF_HOME","/workspace/hf")
from datasets import load_dataset
from vllm import LLM, SamplingParams
MODEL=os.environ["EVAL_MODEL"]; TASK=os.environ.get("EVAL_TASK","gsm8k")
N=int(os.environ.get("EVAL_N","80")); MAXT=int(os.environ.get("EVAL_MAXT","640"))
SPEC=["<eos>","<pad>","<turn|>","<|turn>","<|think|>","<|channel>"]
def final_channel(raw):
    s=raw
    if "<channel|>" in s: s=s.split("<channel|>")[-1]
    for x in SPEC+["<channel|>"]: s=s.replace(x,"")
    return s
def last_num(s):
    s=s.replace(",",""); m=re.findall(r"-?\d+\.?\d*",s); return m[-1] if m else None
def boxed(s):
    i=s.rfind("\\boxed")
    if i<0: return None
    i+=6
    while i<len(s) and s[i]!="{": i+=1
    if i>=len(s): return None
    d=0; st=i+1; j=i
    while j<len(s):
        if s[j]=="{": d+=1
        elif s[j]=="}":
            d-=1
            if d==0: return s[st:j]
        j+=1
    return None
def extract_num(raw):
    a=final_channel(raw); b=boxed(a)
    if b:
        n=last_num(b)
        if n: return n
    for pat in [r"final answer is[:\s\$]*\**(-?\d[\d,]*\.?\d*)",r"answer is[:\s\$]*\**(-?\d[\d,]*\.?\d*)"]:
        m=list(re.finditer(pat,a,re.I))
        if m: return m[-1].group(1).replace(",","")
    return last_num(a)
try:
    from math_verify import parse, verify; HASMV=True
except Exception: HASMV=False
def norm(x):
    if x is None: return None
    x=str(x).strip(); x=re.sub(r"\\text\{([^}]*)\}",r"\1",x); x=x.replace("\\left","").replace("\\right","")
    x=x.replace(" ","").replace("$","").rstrip("."); return x.lower()
def math_eq(p,g):
    if p is None: return False
    if HASMV:
        try:
            if bool(verify(parse(g),parse(p))): return True
        except Exception: pass
    return norm(p)==norm(g)
if TASK=="gsm8k":
    ds=load_dataset("openai/gsm8k","main",split="test").select(range(N))
    items=[{"q":r["question"],"gold":r["answer"].split("####")[-1].strip().replace(",","")} for r in ds]
    mkp=lambda it: it["q"]+"\n\nSolve step by step. End your answer with 'The final answer is \\boxed{...}'."
    def score(it,raw):
        p=extract_num(raw)
        try: return abs(float(p)-float(it["gold"]))<1e-4
        except Exception: return str(p)==str(it["gold"])
else:
    ds=load_dataset("HuggingFaceH4/MATH-500",split="test").select(range(N))
    items=[{"q":r["problem"],"gold":r["answer"]} for r in ds]
    mkp=lambda it: it["q"]+"\n\nSolve concisely. Put ONLY the final answer in \\boxed{} at the end."
    def score(it,raw):
        a=final_channel(raw); b=boxed(a); pred=b if b else a.strip().split("\n")[-1]
        return math_eq(pred,it["gold"])
print(f">>> loading {MODEL} via vLLM", flush=True)
t0=time.time()
llm=LLM(model=MODEL, trust_remote_code=True, max_num_seqs=4, gpu_memory_utilization=0.9,
        hf_overrides={"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1})
print(f">>> loaded in {time.time()-t0:.0f}s", flush=True)
convs=[[{"role":"user","content":mkp(it)}] for it in items]
sp=SamplingParams(max_tokens=MAXT, skip_special_tokens=False)
t1=time.time()
try:
    outs=llm.chat(convs, sp, chat_template_kwargs={"enable_thinking":False})
except TypeError:
    outs=llm.chat(convs, sp)
dt=time.time()-t1
tok=sum(len(o.outputs[0].token_ids) for o in outs)
corr=sum(int(score(it,o.outputs[0].text)) for it,o in zip(items,outs))
print(f"VLLM_RESULT model={MODEL} task={TASK} n={len(items)} acc={corr/len(items):.4f} gen_time={dt:.1f}s out_tok={tok} tok_s={tok/dt:.1f}", flush=True)
