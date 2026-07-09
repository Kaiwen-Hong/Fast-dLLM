import time, torch
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor
MODEL = "google/diffusiongemma-26B-A4B-it"
t0 = time.time()
proc = AutoProcessor.from_pretrained(MODEL)
model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL, dtype="auto", device_map="auto").eval()
print(f"LOADED {time.time()-t0:.1f}s dtype={model.dtype} GPU={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

def run(q, max_new_tokens=256, steps=48, think=False):
    sys = [{"role":"system","content":"<|think|>"}] if think else []
    msg = sys + [{"role":"user","content":q}]
    inp = proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    n_in = inp["input_ids"].shape[1]
    t1=time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, max_denoising_steps=steps)
    dt=time.time()-t1
    seq = out.sequences
    gen = seq[0][n_in:]
    tpf = getattr(out,"tokens_per_forward",None)
    print("="*60, flush=True)
    print(f"Q: {q}", flush=True)
    print(f"gen_tokens={gen.shape[0]} time={dt:.1f}s tokens_per_forward={tpf}", flush=True)
    print("RAW  :", repr(proc.decode(gen, skip_special_tokens=False)[:600]), flush=True)
    print("CLEAN:", proc.decode(gen, skip_special_tokens=True)[:600], flush=True)

run("Why is the sky blue? Answer in two sentences.")
run("What is 17 * 23? Reply with just the number.")
run("Solve: If a train travels 60 km in 1.5 hours, what is its average speed in km/h?", think=True)
print("SMOKE2_DONE", flush=True)
