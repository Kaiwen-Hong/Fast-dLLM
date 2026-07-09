import time, json, os, torch
import transformers.models.diffusion_gemma.generation_diffusion_gemma as gg
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor
MODEL="google/diffusiongemma-26B-A4B-it"
OUT="/workspace/ddgemma/traj"

proc = AutoProcessor.from_pretrained(MODEL)
tok = proc.tokenizer
model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL, dtype="auto", device_map="auto").eval()
model.generation_config.disable_compile = True
print("LOADED", flush=True)

TRAJ=[]
orig_accept = gg.EntropyBoundSampler.accept_canvas
def patched_accept(self, current_canvas, denoiser_canvas, logits, cur_step):
    accepted = orig_accept(self, current_canvas, denoiser_canvas, logits, cur_step)
    with torch.no_grad():
        lf = logits.float()
        ent = torch.distributions.Categorical(logits=lf).entropy()   # (b,L)
        probs = torch.softmax(lf, dim=-1)
        maxp, argmax = probs.max(dim=-1)                              # (b,L)
        mask = self.accepted_token_mask                              # (b,L) bool
    b=0
    step = int(cur_step.item()) if torch.is_tensor(cur_step) else int(cur_step)
    TRAJ.append({
        "step": step,
        "entropy": [round(x,4) for x in ent[b].tolist()],
        "max_prob": [round(x,4) for x in maxp[b].tolist()],
        "current_ids": current_canvas[b].tolist(),
        "argmax_ids": argmax[b].tolist(),
        "accepted_mask": [bool(x) for x in mask[b].tolist()],
        "n_accepted": int(mask[b].sum().item()),
    })
    return accepted
gg.EntropyBoundSampler.accept_canvas = patched_accept

def capture(name, q, max_new_tokens=256, steps=48, think=False):
    global TRAJ
    TRAJ=[]
    msg = ([{"role":"system","content":"<|think|>"}] if think else []) + [{"role":"user","content":q}]
    inp = proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    n_in = inp["input_ids"].shape[1]
    t0=time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, max_denoising_steps=steps)
    dt=time.time()-t0
    seq=out.sequences; gen=seq[0][n_in:]
    final_text = proc.decode(gen, skip_special_tokens=True)
    # build id->token string map for all ids that appear
    ids=set()
    for s in TRAJ:
        ids.update(s["current_ids"]); ids.update(s["argmax_ids"])
    id2tok={int(i): tok.decode([int(i)]) for i in ids}
    rec={"name":name,"question":q,"think":think,"canvas_length":len(TRAJ[0]["entropy"]) if TRAJ else 0,
         "n_steps_taken":len(TRAJ),"max_denoising_steps":steps,"time_s":round(dt,2),
         "tokens_per_forward":float(out.tokens_per_forward[0].item()) if out.tokens_per_forward is not None else None,
         "final_text":final_text,"prompt_len":int(n_in),
         "steps":TRAJ,"id2tok":id2tok}
    p=os.path.join(OUT,name+".json")
    json.dump(rec, open(p,"w"))
    print(f"[{name}] steps={len(TRAJ)} time={dt:.1f}s tpf={rec['tokens_per_forward']:.1f} answer={final_text[:80]!r} -> {p}", flush=True)

capture("sky","Why is the sky blue? Answer in two sentences.")
capture("mult","What is 17 * 23? Reply with only the number.")
capture("count","List the first 6 prime numbers separated by commas.")
capture("math_think","A shop sells pens at 3 for $2. How much do 12 pens cost? Show brief reasoning then the answer.", max_new_tokens=256, think=True)
print("CAPTURE_DONE", flush=True)
