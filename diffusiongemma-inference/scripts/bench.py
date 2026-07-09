#!/usr/bin/env python3
"""DiffusionGemma benchmark harness: GSM8K, MATH-500, MBPP, Sudoku.
Batched generation via HF transformers. Saves per-task JSON with samples + accuracy.
Usage: python3 bench.py --task gsm8k --n 300 --bs 16 --max_new_tokens 512
"""
import argparse, json, os, re, time, random, subprocess, tempfile, textwrap, sys
import torch
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor

MODEL = os.environ.get("DDG_MODEL", "google/diffusiongemma-26B-A4B-it")
OUT_DIR = "/workspace/ddgemma/results"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------- model -----------------------------
_MODEL = None; _PROC = None
def get_model():
    global _MODEL, _PROC
    if _MODEL is None:
        t0 = time.time()
        _PROC = AutoProcessor.from_pretrained(MODEL)
        _PROC.tokenizer.padding_side = "left"
        _MODEL = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL, dtype="auto", device_map="auto").eval()
        _MODEL.generation_config.disable_compile = True
        print(f"[load] {time.time()-t0:.1f}s dtype={_MODEL.dtype} GPU={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
    return _MODEL, _PROC

def batched_generate(prompts, think, max_new_tokens, steps):
    """prompts: list[str] user messages. Returns list of raw decoded generations (special tokens kept)."""
    model, proc = get_model()
    convs = []
    for q in prompts:
        msg = ([{"role":"system","content":"<|think|>"}] if think else []) + [{"role":"user","content":q}]
        convs.append(msg)
    enc = proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True,
                                   padding=True, return_dict=True, return_tensors="pt").to(model.device)
    n_in = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, max_denoising_steps=steps)
    seq = out.sequences
    tpf = out.tokens_per_forward
    res = []
    for i in range(seq.shape[0]):
        gen_ids = seq[i][n_in:]
        raw = proc.decode(gen_ids, skip_special_tokens=False)
        res.append(raw)
    tpf_list = [float(x) for x in tpf.tolist()] if tpf is not None else [None]*seq.shape[0]
    return res, tpf_list

# ----------------------------- answer extraction -----------------------------
SPECIALS = ["<eos>","<pad>","<turn|>","<|turn>","<end_of_turn>","<start_of_turn>","<|think|>"]
def final_channel(raw):
    """Return the text after the last thought-channel close marker, cleaned of special tokens."""
    s = raw
    if "<channel|>" in s:
        s = s.split("<channel|>")[-1]
    for sp in SPECIALS:
        s = s.replace(sp, "")
    # also drop any stray channel-open remnants
    s = s.replace("<|channel>","").strip()
    return s

def full_clean(raw):
    s = raw
    for sp in SPECIALS + ["<channel|>","<|channel>"]:
        s = s.replace(sp, "")
    return s.strip()

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
def last_number(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    return nums[-1] if nums else None

def extract_final_number(raw):
    """Robustly pull the final numeric answer from a (thinking) generation."""
    ans = final_channel(raw)
    has_close = "<channel|>" in raw
    b = extract_boxed(ans)
    if b is not None:
        n = last_number(b)
        if n is not None:
            return n
    for pat in [r"final answer is[:\s\$]*\**(-?\d[\d,]*\.?\d*)",
                r"answer is[:\s\$]*\**(-?\d[\d,]*\.?\d*)",
                r"answer[:\s]*\**\$?(-?\d[\d,]*\.?\d*)"]:
        m = list(re.finditer(pat, ans, re.I))
        if m:
            return m[-1].group(1).replace(",","")
    # else last number of the answer channel (post-<channel|>); if channel never closed, this is thought
    return last_number(ans)

def extract_boxed(s):
    idx = s.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len("\\boxed")
    while i < len(s) and s[i] != "{":
        i += 1
    if i >= len(s):
        return None
    depth = 0; start = i+1; j = i
    while j < len(s):
        if s[j] == "{": depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[start:j]
        j += 1
    return None

def norm_math(x):
    if x is None: return None
    x = str(x).strip()
    x = re.sub(r"\\text\{([^}]*)\}", r"\1", x)
    x = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", x)
    x = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", x)
    x = x.replace("\\left","").replace("\\right","")
    x = x.replace("\\!","").replace("\\,","").replace("\\;","").replace("\\ ","")
    x = x.replace("^\\circ","").replace("^{\\circ}","").replace("\\circ","")
    x = x.replace("\\dfrac","\\frac").replace("\\tfrac","\\frac")
    x = x.replace("$","").replace("\\%","").replace("%","")
    x = x.replace(" ","").rstrip(".")
    if x.startswith("{") and x.endswith("}"): x = x[1:-1]
    return x.lower()

# ----------------------------- datasets -----------------------------
def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k","main",split="test")
    items = []
    for r in ds.select(range(min(n,len(ds)))):
        gold = r["answer"].split("####")[-1].strip().replace(",","")
        items.append({"q": r["question"], "gold": gold})
    return items

def load_math500(n):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500",split="test")
    items=[]
    for r in ds.select(range(min(n,len(ds)))):
        items.append({"q": r["problem"], "gold": r["answer"], "level": r.get("level"), "subject": r.get("subject")})
    return items

def load_mbpp(n):
    from datasets import load_dataset
    try:
        ds = load_dataset("google-research-datasets/mbpp","sanitized",split="test")
        pf = "prompt"
    except Exception:
        ds = load_dataset("mbpp","full",split="test"); pf = "text"
    items=[]
    for r in ds.select(range(min(n,len(ds)))):
        items.append({"q": r[pf], "tests": r["test_list"], "setup": r.get("test_setup_code",""), "code": r.get("code","")})
    return items

# ---- Sudoku: generate puzzles programmatically ----
def _gen_full_solution(rng):
    base=3; side=9
    def pattern(r,c): return (base*(r%base)+r//base+c)%side
    def shuffle(s): return rng.sample(s,len(s))
    rBase=range(base)
    rows=[g*base+r for g in shuffle(list(rBase)) for r in shuffle(list(rBase))]
    cols=[g*base+c for g in shuffle(list(rBase)) for c in shuffle(list(rBase))]
    nums=shuffle(list(range(1,side+1)))
    board=[[nums[pattern(r,c)] for c in cols] for r in rows]
    return board
def load_sudoku(n, clues=36, seed=0):
    rng=random.Random(seed); items=[]
    for _ in range(n):
        sol=_gen_full_solution(rng)
        flat=[str(sol[r][c]) for r in range(9) for c in range(9)]
        puzzle=flat[:]
        idx=list(range(81)); rng.shuffle(idx)
        for k in idx[:81-clues]:
            puzzle[k]="0"
        items.append({"q":"".join(puzzle), "gold":"".join(flat), "clues":clues})
    return items

# ----------------------------- prompts -----------------------------
def make_prompt(task, item):
    if task=="gsm8k":
        return f"{item['q']}\n\nSolve step by step. End your answer with 'The final answer is \\boxed{{...}}'.", True
    if task=="math500":
        return f"{item['q']}\n\nSolve concisely. Put ONLY the final answer in \\boxed{{}} at the end.", False
    if task=="mbpp":
        tests = "\n".join(item["tests"][:3])
        return (f"Write a Python function to solve this task:\n{item['q']}\n\n"
                f"Your function must pass these tests:\n{tests}\n\n"
                f"Return only the function inside a ```python``` code block."), False
    if task=="sudoku":
        p=item["q"]; grid="\n".join(" ".join(p[r*9+c] for c in range(9)) for r in range(9))
        return (f"Solve this 9x9 Sudoku. 0 is empty. Fill every empty cell so each row, column and 3x3 box "
                f"contains 1-9.\n{grid}\n\nGive the completed grid as 81 digits in reading order, ending with "
                f"'The answer is <81 digits>'."), True
    raise ValueError(task)

# ----------------------------- scoring -----------------------------
def score_gsm8k(item, raw):
    pred = extract_final_number(raw)
    gold = item["gold"]
    try:
        ok = abs(float(pred)-float(gold))<1e-4
    except Exception:
        ok = (pred is not None) and (str(pred).strip()==str(gold).strip())
    return ok, pred

_MV = None
def math_eq(pred, gold):
    global _MV
    if _MV is None:
        try:
            from math_verify import parse, verify
            _MV = (parse, verify)
        except Exception:
            _MV = False
    if pred is None:
        return False
    if _MV:
        try:
            parse, verify = _MV
            if bool(verify(parse(gold), parse(pred))):
                return True
        except Exception:
            pass
    return norm_math(pred)==norm_math(gold)
def score_math500(item, raw):
    ans = final_channel(raw)
    b = extract_boxed(ans)
    pred = b if b is not None else ans.strip().split("\n")[-1]
    ok = math_eq(pred if pred else "", item["gold"])
    return ok, pred

def score_mbpp(item, raw):
    ans = final_channel(raw)
    m = re.search(r"```(?:python)?\n(.*?)```", ans, re.DOTALL) or re.search(r"```(?:python)?\n(.*?)```", full_clean(raw), re.DOTALL)
    code = m.group(1) if m else ans
    prog = (item.get("setup","") or "") + "\n" + code + "\n" + "\n".join(item["tests"])
    try:
        r = subprocess.run([sys.executable,"-c",prog], capture_output=True, timeout=12, text=True)
        ok = (r.returncode==0)
    except Exception:
        ok = False
    return ok, code[:400]

def score_sudoku(item, raw):
    ans = final_channel(raw)
    m = re.search(r"answer is[:\s]*([0-9]{81})", ans, re.I)
    digits = m.group(1) if m else None
    if digits is None:
        alld = re.sub(r"[^0-9]","", ans)
        digits = alld[-81:] if len(alld)>=81 else alld
    gold = item["gold"]
    exact = (digits==gold)
    partial = sum(1 for a,b in zip(digits.ljust(81),gold) if a==b)/81.0
    return exact, {"pred":digits, "partial":round(partial,3)}

LOADERS = {"gsm8k":load_gsm8k,"math500":load_math500,"mbpp":load_mbpp,"sudoku":load_sudoku}
SCORERS = {"gsm8k":score_gsm8k,"math500":score_math500,"mbpp":score_mbpp,"sudoku":score_sudoku}

def run(task, n, bs, max_new_tokens, steps, tag="", think_override=None):
    items = LOADERS[task](n)
    print(f"[{task}] {len(items)} items bs={bs} max_new_tokens={max_new_tokens} steps={steps} think_override={think_override}", flush=True)
    results=[]; correct=0; tpf_all=[]; t0=time.time()
    for i in range(0, len(items), bs):
        batch = items[i:i+bs]
        prompts=[]; think=False
        for it in batch:
            p, think = make_prompt(task, it)
            prompts.append(p)
        if think_override is not None:
            think = think_override
        try:
            raws, tpfs = batched_generate(prompts, think, max_new_tokens, steps)
        except Exception as e:
            print(f"  [batch {i} ERROR] {e}", flush=True); continue
        for it, raw, tpf in zip(batch, raws, tpfs):
            try:
                ok, pred = SCORERS[task](it, raw)
            except Exception as e:
                ok, pred = False, f"SCORE_ERR:{e}"
            correct += int(ok); tpf_all.append(tpf)
            results.append({"q": it["q"][:200], "gold": it.get("gold"), "pred": pred, "ok": bool(ok),
                            "tpf": tpf, "raw": raw[:1200]})
        done=len(results); acc=correct/max(done,1)
        el=time.time()-t0
        print(f"  {done}/{len(items)} acc={acc:.3f} tpf={sum(x for x in tpf_all if x)/max(len(tpf_all),1):.1f} elapsed={el:.0f}s", flush=True)
        json.dump({"task":task,"n":len(items),"done":done,"accuracy":acc,
                   "avg_tpf":sum(x for x in tpf_all if x)/max(len(tpf_all),1),
                   "max_new_tokens":max_new_tokens,"steps":steps,"results":results},
                  open(os.path.join(OUT_DIR,f"{task}{tag}.json"),"w"))
    acc=correct/max(len(results),1)
    print(f"[{task}] DONE acc={acc:.4f} n={len(results)} time={time.time()-t0:.0f}s", flush=True)
    return acc

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--tag", default="")
    ap.add_argument("--think", choices=["auto","on","off"], default="auto")
    a=ap.parse_args()
    tov = None if a.think=="auto" else (a.think=="on")
    run(a.task, a.n, a.bs, a.max_new_tokens, a.steps, a.tag, think_override=tov)
    print("BENCH_TASK_DONE", flush=True)
