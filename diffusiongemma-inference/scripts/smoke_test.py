import time, torch
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor
MODEL = "google/diffusiongemma-26B-A4B-it"
print("loading processor + model ...", flush=True)
t0 = time.time()
proc = AutoProcessor.from_pretrained(MODEL)
model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL, dtype="auto", device_map="auto")
model.eval()
print(f"LOADED in {time.time()-t0:.1f}s | dtype={model.dtype} | GPU alloc={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
qs = ["Why is the sky blue? Answer in two sentences.",
      "What is 17 * 23? Reply with just the number.",
      "Write a Python function is_prime(n) that returns True if n is prime."]
for q in qs:
    msg = [{"role":"user","content":q}]
    inp = proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    n_in = inp["input_ids"].shape[1]
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=256, max_denoising_steps=48)
    dt = time.time()-t1
    full = proc.decode(out[0], skip_special_tokens=False)
    gen = proc.decode(out[0][n_in:], skip_special_tokens=True)
    print("="*60, flush=True)
    print("Q:", q, flush=True)
    print("A(clean):", gen, flush=True)
    print(f"[gen {dt:.1f}s | in {n_in} tok | out {out.shape[1]-n_in} tok]", flush=True)
print("SMOKE_DONE", flush=True)
