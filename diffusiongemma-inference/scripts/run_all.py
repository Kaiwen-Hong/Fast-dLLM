import bench, json, time
JOBS = [
  ("gsm8k",  dict(n=500, bs=4, max_new_tokens=640,  steps=48, think_override=False)),
  ("mbpp",   dict(n=257, bs=4, max_new_tokens=768,  steps=48, think_override=False)),
  ("math500",dict(n=500, bs=4, max_new_tokens=1280, steps=48, think_override=False)),
  ("sudoku", dict(n=40,  bs=4, max_new_tokens=768,  steps=48, think_override=True)),
]
summary={}
t0=time.time()
for task,kw in JOBS:
    print(f"\n########## START {task} ##########", flush=True)
    try:
        acc = bench.run(task, **kw)
        summary[task]={"acc":acc}
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[{task}] FATAL {e}", flush=True); summary[task]={"error":str(e)}
    json.dump(summary, open("/workspace/ddgemma/results/SUMMARY.json","w"), indent=2)
    print(f"########## {task} SUMMARY {summary[task]} (total {time.time()-t0:.0f}s) ##########", flush=True)
print("ALL_BENCH_DONE", flush=True)
