"""True multi-host (multi-PROCESS) test for train_tpu.run_step's data feeding.

tests/test_harness_fsdp.py runs SINGLE-process (process_count==1, all devices local), so it never
exercises the multi-host path: run_step must assemble a GLOBAL batch from each process's LOCAL grain
shard. The old code did `jax.device_put(host_local_batch, global_sharding)`, which is wrong on a true
2-VM slice (each process only addresses its own devices). The fix uses
`jax.make_array_from_process_local_data`.

This spawns a real 2-process JAX job on CPU (jax.distributed over localhost) and asserts:
  (1) ASSEMBLY: make_array_from_process_local_data places each process's local rows into the correct
      global positions (process p -> global rows [p*B : (p+1)*B]); allgather == arange. Exact.
  (2) HARNESS:  the real train_tpu.run_step completes multi-host with a finite loss (the OLD device_put
      path crashes here; "completes + finite" is the end-to-end proof the fix works).

Run (jax venv; needs localhost networking for the coordinator):
  export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive
  /home/kaiwen/jax-dlm-baseline/.venv/bin/python jax_ddrive/tests/test_multihost_datafeed.py
Prints MULTIHOST_DATAFEED_TEST_PASS on success.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive"
DATA = "/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd"
SPLIT = "train"
VSMALL = 4096
NPROC = 2
LOCAL_DEV = 4            # CPU devices per process -> 8 global
PER_HOST_BATCH = 4       # 1 sample / device


def _shrink(batch, V=VSMALL, image_tok_orig=151655):
    b = dict(batch)
    img = (batch["input_final"] == image_tok_orig)
    inp = (batch["input_final"] % (V - 1)).astype(batch["input_final"].dtype)
    inp[img] = V - 1
    b["input_final"] = inp
    for k in ("labels_final", "original_labels"):
        lab = batch[k].copy(); m = lab != -100; lab[m] = lab[m] % (V - 1); b[k] = lab
    return b


def _worker():
    pid = int(os.environ["PID"]); coord = os.environ["COORD"]; out = os.environ["OUT"]
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={LOCAL_DEV}"
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jax.experimental import multihost_utils
    jax.distributed.initialize(coordinator_address=coord, num_processes=NPROC, process_id=pid)

    res = {"pid": pid, "nproc": int(jax.process_count()), "ndev": int(jax.device_count())}

    # (1) ASSEMBLY: process p contributes global rows [p*B : (p+1)*B]; allgather must == arange.
    B = PER_HOST_BATCH
    mesh = jax.make_mesh((jax.device_count(), 1), ("fsdp", "tp"))
    sharding = NamedSharding(mesh, P("fsdp", None))
    local = np.array([[pid * B + i] * 3 for i in range(B)], dtype=np.float32)   # rows = global idx
    g = jax.make_array_from_process_local_data(sharding, local, (jax.device_count(), 3))
    allg = np.asarray(multihost_utils.process_allgather(g, tiled=True))
    expect = np.array([[r] * 3 for r in range(jax.device_count())], dtype=np.float32)
    res["assembly_ok"] = bool(np.array_equal(allg, expect))

    # (2) HARNESS: real run_step end-to-end multi-host (old device_put path crashes here).
    from ddrive_jax.train import train_tpu as T
    from ddrive_jax.data import grain_pipeline as gp
    T.IMAGE_TOK = VSMALL - 1
    cfg = T.HarnessConfig(proxy=True, dtype=jnp.float32, d_model=64, n_heads=4, n_kv_heads=2,
                          head_dim=32, n_layers=2, mlp_hidden_size=128, vocab_size=VSMALL,
                          mrope_section=(4, 6, 6), n_fsdp=jax.device_count(), n_tp=1,
                          opt="adamw", lr=1e-3, warmup_steps=2, total_steps=5, seed=0)
    h = T.build_harness(cfg)
    ld = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=PER_HOST_BATCH, seed=0,
                             process_index=jax.process_index(), process_count=jax.process_count())
    it = iter(ld)
    losses = []
    for _ in range(3):
        loss, _ = T.run_step(h, _shrink(next(it)))
        losses.append(float(loss))
    res["losses"] = losses
    res["finite"] = bool(all(np.isfinite(losses)))
    json.dump(res, open(out, "w"))


def main():
    coord = f"localhost:{12000 + os.getpid() % 4000}"
    print(f"spawning {NPROC} procs x {LOCAL_DEV} CPU devices (coord {coord})", flush=True)
    with tempfile.TemporaryDirectory() as td:
        outs = [os.path.join(td, f"w{i}.json") for i in range(NPROC)]
        logs = [os.path.join(td, f"w{i}.log") for i in range(NPROC)]
        procs = []
        for i in range(NPROC):
            env = dict(os.environ, ROLE="worker", PID=str(i), COORD=coord, OUT=outs[i],
                       PYTHONPATH=REPO, JAX_PLATFORMS="cpu")
            procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__)],
                         env=env, stdout=open(logs[i], "w"), stderr=subprocess.STDOUT))
        rcs = [p.wait() for p in procs]
        if any(rc != 0 for rc in rcs):
            for i in range(NPROC):
                print(f"--- worker {i} (rc={rcs[i]}) ---")
                print(open(logs[i]).read()[-2500:])
            raise SystemExit(f"worker(s) failed: rcs={rcs}")
        res = [json.load(open(o)) for o in outs]

    assert all(r["nproc"] == NPROC and r["ndev"] == NPROC * LOCAL_DEV for r in res), res
    assert all(r["assembly_ok"] for r in res), f"assembly mismatch: {res}"
    assert all(r["finite"] for r in res), f"non-finite loss: {res}"
    # both processes must observe the SAME global loss (collective psum is consistent)
    assert abs(res[0]["losses"][0] - res[1]["losses"][0]) < 1e-6, \
        f"processes disagree on global loss: {res[0]['losses'][0]} vs {res[1]['losses'][0]}"
    print(f"  [1] ASSEMBLY ok: make_array_from_process_local_data placed rows correctly across "
          f"{NPROC} procs / {NPROC*LOCAL_DEV} devices")
    print(f"  [2] HARNESS ok: run_step multi-host, 3 steps finite, both procs agree: "
          f"losses[p0]={[round(x,4) for x in res[0]['losses']]}")
    print(f"  (process_count={res[0]['nproc']}, global devices={res[0]['ndev']})")
    print("MULTIHOST_DATAFEED_TEST_PASS")


if __name__ == "__main__":
    if os.environ.get("ROLE") == "worker":
        _worker()
    else:
        main()
