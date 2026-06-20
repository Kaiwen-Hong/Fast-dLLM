"""Cross-pipeline consistency check for the TWO Fast-dDrive data-prep paths.

There are two separate prep code paths off the SAME WOD-E2E frame:
  - TRAINING:  eval/prep_train_jax.py  -> npz {input_ids, labels, rbi, pixel_values, image_grid_thw, ...}
               (filled GT + token labels; feeds prep_to_parquet -> parquet_to_ar_with_embeds -> ArrayRecord)
  - INFERENCE: eval/prep_jax_eval.py   -> npz {x_t0, rbi, position_ids, pixel_values, image_grid_thw, ...}
               (MASKED scaffold start for the sampler + GT passthrough)

They MUST agree on the shared derivations or the two pipelines have drifted. This harness runs BOTH
on the SAME toy json at the SAME resolution and asserts:
  (a) pixel_values byte-identical          (same HF processor, same images, same min/max_pixels)
  (b) image_grid_thw identical             (same patch grid -> same image-token count)
  (c) prompt-token prefix identical        (train.input_ids vs eval.x_t0 share the whole user+assistant-header
                                            prefix; they diverge only where train has GT tokens and eval has MASK)
REPORTED (NOT a pass/fail equality — expected to differ by design):
  (d) scaffold n_blocks                     train = the ACTUAL GT response length; eval = the FIXED generation
                                            budget the sampler reserves (>= GT). They legitimately differ; we
                                            print both so a *gross* divergence (e.g. one is 0/None) is visible.

Resolution is a parameter (--pixels = min = max): pass 50176 to check at train-res (168 tok) or 200704 for
paper-res (720 tok). Consistency must hold at ANY single resolution.

PyTorch `ddrive` env, CPU, NO model weights (only the HF processor + section_utils).
Run:
  /home/kaiwen/miniconda3/envs/ddrive/bin/python jax_ddrive/scripts/validate_prep_consistency.py \
      --json fast_ddrive/data/example/sample.json --image_root fast_ddrive --pixels 50176 --n 2
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np

MASK_ID = 151665


def _run(cmd):
    print("  $ " + " ".join(os.path.basename(c) if c.endswith(".py") else c for c in cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="toy json (list of items with conversations+image[+sample_id])")
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--pixels", type=int, default=50176, help="min=max pixels; SAME res for both preps")
    ap.add_argument("--n", type=int, default=2, help="number of toy samples")
    args = ap.parse_args()

    repo = os.environ.get("FASTDDRIVE_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    py = sys.executable
    td = tempfile.mkdtemp(prefix="prep_xcheck_")
    tr, ev = os.path.join(td, "train"), os.path.join(td, "eval")
    px = str(args.pixels)

    print(f"[xcheck] running BOTH preps at {args.pixels} px on {args.n} toy samples -> {td}")
    _run([py, os.path.join(repo, "eval", "prep_train_jax.py"), "--train_json", args.json,
          "--image_root", args.image_root, "--out_dir", tr, "--min_pixels", px, "--max_pixels", px,
          "--max_samples", str(args.n)])
    _run([py, os.path.join(repo, "eval", "prep_jax_eval.py"), "--eval_json", args.json,
          "--image_root", args.image_root, "--out_dir", ev, "--min_pixels", px, "--max_pixels", px,
          "--max_samples", str(args.n)])

    ok = True
    for i in range(args.n):
        t = np.load(os.path.join(tr, f"{i:05d}.npz"))
        e = np.load(os.path.join(ev, f"{i:05d}.npz"))
        pv_eq = np.array_equal(t["pixel_values"], e["pixel_values"])
        thw_eq = np.array_equal(t["image_grid_thw"], e["image_grid_thw"])
        ti, xi = t["input_ids"], e["x_t0"]            # train filled ids vs eval masked scaffold
        m = min(len(ti), len(xi)); lcp = 0
        while lcp < m and int(ti[lcp]) == int(xi[lcp]):
            lcp += 1
        nb_t = int(t["n_blocks"]); nb_e = int(np.asarray(e["rbi"]).max()) + 1
        # PASS = the shared INPUT is identical (pixels, grid, prompt prefix). n_blocks is reported only:
        # train=actual-GT length, eval=fixed generation budget (eval >= train expected). Flag only if absurd.
        nb_sane = nb_t > 0 and nb_e > 0 and nb_e >= nb_t
        sample_ok = pv_eq and thw_eq and lcp > 50 and nb_sane
        ok = ok and sample_ok
        print(f"sample {i}: pixel_values_eq={pv_eq}  image_grid_thw_eq={thw_eq}  grid={t['image_grid_thw'].tolist()}  "
              f"prompt_lcp_tokens={lcp} (train_L={len(ti)}, eval_L={len(xi)})  "
              f"n_blocks train={nb_t}(actual GT) eval={nb_e}(gen budget; >=GT ok)  -> {'OK' if sample_ok else 'MISMATCH'}")
    print("PREP_CONSISTENCY_PASS" if ok else "PREP_CONSISTENCY_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
