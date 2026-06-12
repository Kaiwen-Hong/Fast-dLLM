"""Round-2 semantic verification of the SASD ArrayRecord training dataset.

Verifies, for N sampled records of wod_e2e_sasd_full_tfexample_ar, that the stored
row faithfully encodes [image1, image2, image3, prompt, answer]:

  raw tfrecord ──convert_wod_e2e.py──▶ JSON+JPEG ──prep_train_jax.py──▶ ref npz
                                                      (the EXACT build pipeline, re-run)
  ar record ──parse tf.Example──▶ ar npz
  compare:  every array column bit-exact (ar vs ref)
          + structure self-checks (image_pad runs == grid_thw, labels span, weights, ...)
          + answer-text round-trip (decode(labels span) == process_gpt(original gpt))
          + image reconstruction from ar pixel_values (inverse patchify+denorm) -> PNGs
            next to the original JPEGs, for human review of content + camera order.

Subcommands (run under different envs; see verify_ar_round2.sh):
  locate   (any env w/ pyarrow)        find sample_ids in the packed parquet -> (shard, row)
  extract  (env w/ tf + array_record)  random-access read those ar records -> npz dumps
  compare  (ddrive env)                all checks + per-sample review.md + report.md
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = "/home/kaiwen/Desktop/research/Fast-dLLM"
SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
sys.path.insert(0, REPO + "/jax_ddrive")

ARRAY_DTYPES = {
    "input_ids": "int64", "labels": "int64", "rbi": "int32", "turn": "int32",
    "scaffold": "bool", "weight_vec": "float32", "block_alpha": "float32",
    "block_beta": "float32", "position_ids": "int32", "vision_mask": "bool",
    "pixel_values": "float16", "image_grid_thw": "int64",
}
IM_START, IM_END = 151644, 151645
VSTART, VEND, VPAD, IMAGE_PAD = 151652, 151653, 151654, 151655
MASK_ID, NULL_ID = 151665, 151666
# (alpha, beta) -> section (bijective; mirrors prep_train_jax NOISE_SCHED)
AB_TO_SECTION = {(1.0, 2.0): "critical_objects", (1.0, 1.0): "explanation",
                 (1.0, 1.5): "future_meta_behavior", (2.0, 1.0): "trajectory"}
SECTION_W = {"critical_objects": 1.5, "explanation": 1.0,
             "future_meta_behavior": 2.0, "trajectory": 3.0}


# ──────────────────────────────────────────────────────────────── locate ────
def cmd_locate(args):
    import pyarrow.parquet as pq
    ids = json.load(open(args.ids_json))          # ordered list of sample_ids
    want = set(ids)
    found = {}
    files = sorted(glob.glob(os.path.join(args.packed_dir, "train-*.parquet")))
    assert files, f"no packed parquet under {args.packed_dir}"
    for si, f in enumerate(files):
        col = pq.read_table(f, columns=["sample_id"]).column("sample_id").to_pylist()
        for ri, sid in enumerate(col):
            if sid in want and sid not in found:
                found[sid] = {"shard": si, "row": ri, "file": os.path.basename(f)}
        if len(found) == len(want):
            break
    missing = [s for s in ids if s not in found]
    out = {"order": ids, "locations": found, "missing": missing}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[locate] found {len(found)}/{len(ids)}; missing={missing}")
    assert not missing, "some sample_ids not present in the packed dataset"


# ─────────────────────────────────────────────────────────────── extract ────
def cmd_extract(args):
    import tensorflow as tf
    from array_record.python.array_record_data_source import ArrayRecordDataSource
    loc = json.load(open(args.locations))
    os.makedirs(args.out_dir, exist_ok=True)
    ar_files = sorted(glob.glob(os.path.join(args.ar_dir, "train-*.arrayrecord")))
    for sid in loc["order"]:
        e = loc["locations"][sid]
        path = ar_files[e["shard"]]
        assert os.path.basename(path).startswith(f"train-{e['shard']:05d}-"), path
        ds = ArrayRecordDataSource([path])
        rec = ds[e["row"]]
        ex = tf.train.Example.FromString(rec)
        feat = ex.features.feature
        got_sid = feat["sample_id"].bytes_list.value[0].decode()
        assert got_sid == sid, (
            f"ar row order mismatch: shard {e['shard']} row {e['row']} has sample_id "
            f"{got_sid!r}, expected {sid!r}")
        arrs = {}
        for k, dt in ARRAY_DTYPES.items():
            t = tf.io.parse_tensor(feat[k].bytes_list.value[0], getattr(tf, {
                "int64": "int64", "int32": "int32", "bool": "bool",
                "float32": "float32", "float16": "half"}[dt]))
            arrs[k] = t.numpy()
        arrs["L"] = np.int64(feat["L"].int64_list.value[0])
        arrs["n_blocks"] = np.int64(feat["n_blocks"].int64_list.value[0])
        np.savez(os.path.join(args.out_dir, sid.replace(os.sep, "_") + ".npz"), **arrs)
        print(f"[extract] {sid} <- ar shard {e['shard']} row {e['row']} "
              f"(L={arrs['L']}, pixels={arrs['pixel_values'].shape})")
    print("EXTRACT_DONE")


# ─────────────────────────────────────────────────────────────── compare ────
def _runs(ids):
    """[(token_id, start, length)] of maximal constant runs."""
    out, i, n = [], 0, len(ids)
    while i < n:
        j = i
        while j < n and ids[j] == ids[i]:
            j += 1
        out.append((int(ids[i]), i, j - i))
        i = j
    return out


def _render_collapsed(ids, tok, collapse={IMAGE_PAD, MASK_ID, NULL_ID}):
    """Decode ids to text, collapsing runs of placeholder specials to <name>*N."""
    parts, buf = [], []
    def flush():
        if buf:
            parts.append(tok.decode(buf, skip_special_tokens=False))
            buf.clear()
    for t, s, ln in _runs(list(ids)):
        if t in collapse and ln > 1:
            flush()
            parts.append(f"⟨{tok.decode([t])}×{ln}⟩")
        else:
            buf.extend([t] * ln)
    flush()
    return "".join(parts)


def _inverse_patchify(pv, grid_thw, mean, std, merge=2, tps=2, p=14):
    """ar pixel_values [N,1176] -> list of PIL images (frame 0 of each image)."""
    from PIL import Image
    imgs, off = [], 0
    for (t, gh, gw) in [tuple(int(x) for x in g) for g in grid_thw]:
        n = t * gh * gw
        x = pv[off:off + n].astype(np.float32); off += n
        x = x.reshape(t, gh // merge, gw // merge, merge, merge, 3, tps, p, p)
        x = x.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8)      # inverse of (0,3,6,4,7,2,1,5,8)
        x = x.reshape(t * tps, 3, gh * p, gw * p)[0]     # temporal frame 0 -> [C,H,W]
        x = (x * std[:, None, None] + mean[:, None, None]) * 255.0
        x = np.clip(np.rint(x.transpose(1, 2, 0)), 0, 255).astype(np.uint8)
        imgs.append(Image.fromarray(x))
    return imgs


def cmd_compare(args):
    from PIL import Image
    from transformers import AutoTokenizer
    sys.path.insert(0, REPO + "/jax_ddrive/eval")
    from prep_train_jax import process_gpt                      # the build's normalizer
    from ddrive_jax.diffusion import noise as noise_mod         # online SASD noising

    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    pcfg = json.load(open(os.path.join(SNAP, "preprocessor_config.json")))
    mean = np.array(pcfg["image_mean"], np.float32)
    std = np.array(pcfg["image_std"], np.float32)

    src = json.load(open(args.src_json))
    src_by_id = {s["sample_id"]: s for s in src}
    ref_man = {m["sample_id"]: m["npz"]
               for m in json.load(open(os.path.join(args.ref_npz, "manifest.json")))}
    loc = json.load(open(args.locations))
    os.makedirs(args.out_dir, exist_ok=True)

    summary = []
    for k, sid in enumerate(loc["order"]):
        ar = dict(np.load(os.path.join(args.ar_npz, sid.replace(os.sep, "_") + ".npz")))
        ref = dict(np.load(os.path.join(args.ref_npz, ref_man[sid])))
        item = src_by_id[sid]
        sdir = os.path.join(args.out_dir, f"{k:02d}_{sid}")
        os.makedirs(sdir, exist_ok=True)
        checks = {}

        # 1) column-exact: ar vs freshly re-encoded reference
        for col in ARRAY_DTYPES:
            checks[f"col:{col}"] = bool(np.array_equal(ar[col], ref[col]))
        checks["col:L"] = int(ar["L"]) == int(ref["input_ids"].shape[0])
        checks["col:n_blocks"] = int(ar["n_blocks"]) == int(ref["n_blocks"])
        pv_diff = float(np.abs(ar["pixel_values"].astype(np.float32)
                               - ref["pixel_values"].astype(np.float32)).max()) \
            if ar["pixel_values"].shape == ref["pixel_values"].shape else float("nan")

        ids = ar["input_ids"]; labels = ar["labels"]; L = int(ids.shape[0])
        grid = ar["image_grid_thw"]

        # 2) structure self-checks (independent of the re-encode)
        checks["struct:L%32==0"] = (L % 32 == 0)
        pad_runs = [(s, ln) for t, s, ln in _runs(list(ids)) if t == IMAGE_PAD]
        exp_runs = [int(t * h * w) // 4 for (t, h, w) in grid]
        checks["struct:n_image_runs==n_images"] = (len(pad_runs) == len(grid) == 3)
        checks["struct:run_lengths==grid_thw/4"] = ([ln for _, ln in pad_runs] == exp_runs)
        checks["struct:sum_runs==n_patches/4"] = (
            sum(ln for _, ln in pad_runs) == int(ar["pixel_values"].shape[0]) // 4)
        vm_ref = np.isin(ids, [IMAGE_PAD, VSTART, VPAD])
        checks["struct:vision_mask_formula"] = bool(np.array_equal(ar["vision_mask"], vm_ref))
        resp = np.where(labels != -100)[0]
        rs, re_ = int(resp[0]), int(resp[-1]) + 1
        checks["struct:labels_after_assistant"] = bool(
            ids[rs - 3] == IM_START and ids[rs - 2] == 77091 and ids[rs - 1] == 198)
        checks["struct:labels_end_at_im_end"] = bool(
            ids[re_ - 1] == IM_END and np.array_equal(labels[rs:re_], ids[rs:re_]))
        checks["struct:weights_in_vocab"] = bool(
            set(np.unique(ar["weight_vec"]).tolist()) <= {1.0, 1.5, 2.0, 3.0})

        # 3) answer-text round-trip: decode(labels span) == process_gpt(original gpt)
        gpt_norm = process_gpt(item["conversations"][1]["value"], tok)
        ans_decoded = tok.decode(ids[rs:re_ - 1], skip_special_tokens=False)
        checks["text:answer==process_gpt(orig)"] = (ans_decoded == gpt_norm)

        # 3b) the encoded trajectory must be the real GT @1s (±0.005 = 2-decimal rounding)
        import re as re_mod
        ans_obj = json.loads(ans_decoded)
        enc_wp = [[float(a), float(b)] for a, b in re_mod.findall(
            r'([+-]\d+\.\d+),\s*([+-]\d+\.\d+)', ans_obj["trajectory"])]
        gt5 = item.get("gt_5waypoints_1s", [])
        checks["traj:encoded_5wp==GT@1s(±0.005)"] = (
            len(enc_wp) == len(gt5) == 5 and
            float(np.abs(np.array(enc_wp) - np.array(gt5)).max()) <= 0.0051)
        # ego history: the numeric payload inside the prompt text
        hist = [[float(t), float(x), float(y)] for t, x, y in re_mod.findall(
            r'\(t([+-][\d.]+)s\) \[([-\d.]+), ([-\d.]+)\]',
            item["conversations"][0]["value"])]
        checks["prompt:7_history_points"] = (len(hist) == 7)

        # 4) image reconstruction from ar pixel_values -> PNGs + side-by-side
        recon = _inverse_patchify(ar["pixel_values"], grid, mean, std)
        tags = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"]
        rows = []
        for i, im in enumerate(recon):
            im.save(os.path.join(sdir, f"recon_{i}_{tags[i]}.png"))
            orig = Image.open(os.path.join(args.image_root, item["image"][i])).convert("RGB")
            rows.append((orig.resize(im.size, Image.BICUBIC), im))
        w, h = recon[0].size
        sheet = Image.new("RGB", (2 * w + 12, 3 * h + 24), "white")
        for i, (o, r) in enumerate(rows):
            sheet.paste(o, (0, i * (h + 8)))
            sheet.paste(r, (w + 12, i * (h + 8)))
        sheet.save(os.path.join(sdir, "side_by_side__orig_vs_recon.png"))

        # 5) per-section table from (alpha,beta) -> section
        ba, bb, rbi, wv = ar["block_alpha"], ar["block_beta"], ar["rbi"], ar["weight_vec"]
        sec_rows = []
        for b in range(int(ar["n_blocks"])):
            sec = AB_TO_SECTION.get((round(float(ba[b]), 4), round(float(bb[b]), 4)), "?")
            pos = np.where(rbi == b)[0]
            wset = sorted(set(np.round(wv[pos], 2).tolist())) if pos.size else []
            ok = (len(wset) == 1 and wset[0] == SECTION_W.get(sec))
            sec_rows.append((b, sec, int(pos.size), wset, float(ba[b]), float(bb[b]), ok))
        checks["struct:block_weights_match_section"] = all(r[-1] for r in sec_rows)

        # 6) one-step online-noising preview (what the train step actually consumes)
        ifn, lfn, _ol, _w = noise_mod.make_batch(
            {kk: ref[kk] for kk in ("input_ids", "labels", "rbi", "scaffold",
                                    "weight_vec", "block_alpha", "block_beta")},
            np.random.default_rng(0))
        n_masked = int((ifn[0, :L] == MASK_ID).sum())

        # 7) dump everything the viz step (autovla env, matplotlib) needs
        fmb = ans_obj.get("future_meta_behavior", {})
        json.dump({
            "sample_id": sid, "L": L, "rs": rs, "re": re_,
            "nav": item.get("navigation_command", "?"),
            "fmb_longitudinal": str(fmb.get("longitudinal", "")).replace("<|NULL|>", ""),
            "fmb_lateral": str(fmb.get("lateral", "")).replace("<|NULL|>", ""),
            "history_t_x_y": hist,
            "encoded_5wp": enc_wp,
            "gt_5wp_1s": gt5,
            "gt_20wp_4hz": item.get("future waypoints", []),
            "block_to_section": {str(b): sec for b, sec, *_ in sec_rows},
        }, open(os.path.join(sdir, "viz_data.json"), "w"), indent=2)

        ok_all = all(checks.values())
        summary.append((sid, ok_all, checks, pv_diff))

        # ── per-sample review.md ──
        with open(os.path.join(sdir, "review.md"), "w") as f:
            f.write(f"# sample {k:02d} — `{sid}`  →  {'✅ ALL PASS' if ok_all else '❌ FAILURES'}\n\n")
            f.write(f"- ar location: shard {loc['locations'][sid]['shard']}, "
                    f"row {loc['locations'][sid]['row']} of "
                    f"`{loc['locations'][sid]['file']}`\n")
            f.write(f"- L = {L} (pad to %32), n_blocks = {int(ar['n_blocks'])}, "
                    f"prompt len = {rs - 3}, answer span = [{rs}, {re_}), "
                    f"MASK tail = {int((ids[re_:] == MASK_ID).sum())}\n")
            f.write(f"- image_grid_thw = {grid.tolist()}  → image_pad runs "
                    f"{[ln for _, ln in pad_runs]} (merged tokens/img), "
                    f"pixel_values {tuple(ar['pixel_values'].shape)} fp16\n")
            f.write(f"- max|ar.pixel_values − re-encoded.pixel_values| = {pv_diff:g}\n\n")
            f.write("## visual review files (this directory)\n\n"
                    "| file | shows |\n|---|---|\n"
                    "| `side_by_side__orig_vs_recon.png` | original JPEGs vs images reconstructed from the dataset's `pixel_values` (content + camera order) |\n"
                    "| `bev_trajectory.png` | BEV: ego history from the prompt, encoded 5-waypoint answer, GT 20-pt future |\n"
                    "| `sequence_layout.png` | what each of the L token positions is (prompt / vision / section / scaffold / pad) |\n"
                    "| `position_ids.png` | 3D M-RoPE position channels (t/h/w) over the sequence |\n"
                    "| `attn_mask.png` | the training hybrid block-causal attention mask built from rbi/turn |\n\n")
            f.write("## checks\n\n| check | result |\n|---|---|\n")
            for name, ok in checks.items():
                f.write(f"| {name} | {'✅' if ok else '❌'} |\n")
            f.write("\n## sections (block → section via (α,β); weight per token)\n\n")
            f.write("| block | section | #value-pos | weights | α | β | ok |\n|---|---|---|---|---|---|---|\n")
            for b, sec, np_, wset, a, bv, ok in sec_rows:
                f.write(f"| {b} | {sec} | {np_} | {wset} | {a:g} | {bv:g} | {'✅' if ok else '❌'} |\n")
            f.write("\n## original prompt (stage-1 JSON, `conversations[0]`)\n\n```\n"
                    + item["conversations"][0]["value"] + "\n```\n")
            f.write("\n## original gpt target → after `process_gpt` normalization\n\n```\n"
                    + gpt_norm + "\n```\n")
            f.write("\n## decoded `input_ids` (placeholders collapsed)\n\n```\n"
                    + _render_collapsed(ids, tok) + "\n```\n")
            f.write("\n## decoded labels span (the trained answer)\n\n```\n"
                    + ans_decoded + "\n```\n")
            f.write(f"\n## online-noising preview (seed 0): one SASD train step masks "
                    f"{n_masked}/{int((labels != -100).sum())} response tokens\n\n"
                    "noisy-half excerpt around trajectory:\n\n```\n")
            traj_blocks = [b for b, sec, *_ in sec_rows if sec == "trajectory"]
            if traj_blocks:
                tpos = np.where(np.isin(rbi, traj_blocks))[0]
                s0, e0 = max(int(tpos[0]) - 4, 0), min(int(tpos[-1]) + 5, L)
                f.write(_render_collapsed(ifn[0, s0:e0], tok) + "\n")
            f.write("```\n")
        print(f"[compare] {k:02d} {sid}: {'PASS' if ok_all else 'FAIL'} "
              f"(pv_diff={pv_diff:g})")

    # ── top-level report ──
    n_pass = sum(1 for _, ok, _, _ in summary if ok)
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("# Round-2 semantic verification — wod_e2e_sasd_full_tfexample_ar\n\n")
        f.write(f"**{n_pass}/{len(summary)} samples pass every check.**\n\n")
        f.write("Pipeline re-run for these samples: raw tfrecord → `convert_wod_e2e.py "
                "--with_target` → `prep_train_jax.py` (the exact build chain), then every "
                "array column compared bit-exact against the ArrayRecord rows.\n\n")
        f.write("## encoding layout (what `input_ids` actually is)\n\n```\n"
                "input_ids = [ chat-template header,\n"
                "              <|vision_start|> <|image_pad|>×n1 <|vision_end|>   ← image1 (FRONT_LEFT)\n"
                "              <|vision_start|> <|image_pad|>×n2 <|vision_end|>   ← image2 (FRONT)\n"
                "              <|vision_start|> <|image_pad|>×n3 <|vision_end|>   ← image3 (FRONT_RIGHT)\n"
                "              prompt text ... <|im_end|>\n"
                "              <|im_start|>assistant\\n {JSON answer, normalized} <|im_end|>\n"
                "              |<MASK>|×pad  (to a multiple of 32) ]\n"
                "image content is NOT in input_ids: the <|image_pad|> tokens are constant\n"
                "placeholders; pixels live in pixel_values [N,1176] (14×14×3×2 patches,\n"
                "CLIP-normalized) and are scattered in at forward time. labels = answer span\n"
                "only (-100 elsewhere); weight_vec/rbi/α,β carry the per-section SASD schedule.\n```\n\n")
        f.write("| # | sample_id | result | failed checks | max px diff |\n|---|---|---|---|---|\n")
        for i, (sid, ok, checks, pvd) in enumerate(summary):
            bad = [n for n, v in checks.items() if not v]
            f.write(f"| {i:02d} | `{sid}` | {'✅' if ok else '❌'} | "
                    f"{', '.join(bad) if bad else '—'} | {pvd:g} |\n")
        f.write("\nPer-sample packets (review.md + original JPEGs vs reconstructed PNGs) "
                "are in the numbered subdirectories.\n")
    print(f"\n[compare] {n_pass}/{len(summary)} PASS → {args.out_dir}/report.md")
    sys.exit(0 if n_pass == len(summary) else 1)


# ─────────────────────────────────────────────────────────────────── viz ────
def _hybrid_mask_np(rbi, turn):
    """numpy port of diffusion/masks.hybrid_block_causal_mask_dense -> [2n,2n] bool."""
    n = rbi.shape[0]
    idx = np.arange(2 * n)
    x0 = idx >= n
    pos = np.where(x0, idx - n, idx)
    trn = turn[pos]
    x0q, x0k = x0[:, None], x0[None, :]
    tq, tk = trn[:, None], trn[None, :]
    pq, pk = pos[:, None], pos[None, :]
    return ((~x0q) & (~x0k) & (tq == tk)) | ((tq > tk) & x0k & (~x0q)) \
        | (x0q & x0k & (pq >= pk))


SEC_ORDER = ["critical_objects", "explanation", "future_meta_behavior", "trajectory"]


def cmd_viz(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    loc = json.load(open(args.locations))
    for k, sid in enumerate(loc["order"]):
        sdir = os.path.join(args.out_dir, f"{k:02d}_{sid}")
        vd = json.load(open(os.path.join(sdir, "viz_data.json")))
        ar = dict(np.load(os.path.join(args.ar_npz, sid.replace(os.sep, "_") + ".npz")))
        ids, rbi, turn = ar["input_ids"], ar["rbi"], ar["turn"]
        L, rs, re_ = vd["L"], vd["rs"], vd["re"]
        b2s = {int(b): s for b, s in vd["block_to_section"].items()}

        # ── 1) BEV trajectory: history (prompt) + encoded answer + GT ──
        h = np.array(vd["history_t_x_y"], np.float64)
        enc = np.array(vd["encoded_5wp"], np.float64)
        gt20 = np.array(vd["gt_20wp_4hz"], np.float64)
        fig, axx = plt.subplots(figsize=(8, 5.5))
        if gt20.size:
            axx.plot(gt20[:, 0], gt20[:, 1], "-", c="0.65", lw=2, marker=".", ms=5,
                     label="GT future (20 pt @4 Hz, tfrecord)")
        axx.plot(enc[:, 0], enc[:, 1], "x", c="crimson", ms=11, mew=2.5,
                 label="encoded answer trajectory (5 wp @1 s)")
        if h.size:
            axx.plot(h[:, 1], h[:, 2], "-o", c="royalblue", ms=5,
                     label="ego history in prompt (7 pt @0.5 s)")
        axx.plot([0], [0], "k*", ms=14, label="ego (t=0)")
        axx.set_xlabel("x forward (m)")
        axx.set_ylabel("y left (m)")
        axx.set_title(f"{sid}\nnav={vd['nav']}  fmb: {vd['fmb_longitudinal']} / "
                      f"{vd['fmb_lateral']}")
        axx.axis("equal")
        axx.grid(alpha=0.3)
        axx.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(sdir, "bev_trajectory.png"), dpi=120)
        plt.close(fig)

        # ── 2) sequence layout strips ──
        # row 1: region/section.  row 2: training role (prompt / scaffold / value / pad)
        cat = np.zeros(L, np.int32)                                   # 0 = prompt text
        cat[np.isin(ids, [VSTART, VEND, VPAD, IMAGE_PAD])] = 1        # vision span
        for b, sec in b2s.items():
            cat[rbi == b] = 2 + SEC_ORDER.index(sec)                  # 2..5 sections
        ans_other = (np.arange(L) >= rs) & (np.arange(L) < re_) & (rbi < 0)
        cat[ans_other] = 6                                            # answer boundary
        cat[re_:] = 7                                                 # MASK pad tail
        role = np.zeros(L, np.int32)                                  # 0 prompt/-100
        resp = ar["labels"] != -100
        role[resp & ar["scaffold"]] = 1                               # frozen scaffold
        role[resp & ~ar["scaffold"]] = 2                              # value (denoised)
        role[re_:] = 3                                                # pad
        c1 = ListedColormap(["#d9d9d9", "#7fb3d5", "#f5b041", "#82e0aa",
                             "#bb8fce", "#e74c3c", "#5d6d7e", "#17202a"])
        c2 = ListedColormap(["#d9d9d9", "#5d6d7e", "#e74c3c", "#17202a"])
        fig, axes = plt.subplots(2, 1, figsize=(14, 2.6), sharex=True)
        axes[0].imshow(cat[None], aspect="auto", cmap=c1, vmin=0, vmax=7,
                       interpolation="nearest")
        axes[1].imshow(role[None], aspect="auto", cmap=c2, vmin=0, vmax=3,
                       interpolation="nearest")
        axes[0].set_yticks([]); axes[1].set_yticks([])
        axes[1].set_xlabel(f"token position (L={L})")
        axes[0].legend(handles=[
            Patch(fc="#d9d9d9", label="prompt text"), Patch(fc="#7fb3d5", label="vision span"),
            Patch(fc="#f5b041", label="critical_objects (w1.5)"),
            Patch(fc="#82e0aa", label="explanation (w1.0)"),
            Patch(fc="#bb8fce", label="future_meta_behavior (w2.0)"),
            Patch(fc="#e74c3c", label="trajectory (w3.0)"),
            Patch(fc="#5d6d7e", label="answer boundary"), Patch(fc="#17202a", label="MASK pad")],
            ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 2.2))
        axes[1].legend(handles=[
            Patch(fc="#d9d9d9", label="no loss (prompt)"),
            Patch(fc="#5d6d7e", label="scaffold (frozen)"),
            Patch(fc="#e74c3c", label="value (noised+trained)"),
            Patch(fc="#17202a", label="pad")], ncol=4, fontsize=7, loc="upper center",
            bbox_to_anchor=(0.5, -0.55))
        fig.subplots_adjust(top=0.62, bottom=0.30, left=0.02, right=0.99)
        fig.savefig(os.path.join(sdir, "sequence_layout.png"), dpi=120)
        plt.close(fig)

        # ── 3) 3D M-RoPE position_ids ──
        pos = ar["position_ids"]
        fig, axx = plt.subplots(figsize=(10, 3.2))
        for ch, name, c in zip(range(3), ["temporal", "height", "width"],
                               ["#1f77b4", "#2ca02c", "#d62728"]):
            axx.plot(pos[ch], lw=1.0, label=name, c=c, alpha=0.85)
        axx.set_xlabel("token position")
        axx.set_ylabel("M-RoPE position id")
        axx.set_title("3D M-RoPE position_ids — text: 3 identical diagonals; "
                      "image spans: t flat, h/w sawtooth")
        axx.legend(fontsize=8)
        axx.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(sdir, "position_ids.png"), dpi=120)
        plt.close(fig)

        # ── 4) training attention mask (doubled hybrid block-causal) ──
        m = _hybrid_mask_np(rbi, turn)
        ds = m[::4, ::4]
        fig, axx = plt.subplots(figsize=(6.4, 6.4))
        axx.imshow(ds, cmap="gray_r", interpolation="nearest")
        half = ds.shape[0] // 2
        axx.axhline(half - 0.5, c="crimson", lw=0.8)
        axx.axvline(half - 0.5, c="crimson", lw=0.8)
        axx.set_title("training attention mask [2L,2L] (dark=attend, ×4 downsample)\n"
                      "quadrants: TL noisy↔noisy, TR noisy→clean, BR clean causal")
        axx.set_xticks([]); axx.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(sdir, "attn_mask.png"), dpi=120)
        plt.close(fig)
        print(f"[viz] {k:02d} {sid}: bev/layout/positions/mask written")
    print("VIZ_DONE")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("locate")
    p.add_argument("--packed_dir", required=True)
    p.add_argument("--ids_json", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--ar_dir", required=True)
    p.add_argument("--locations", required=True)
    p.add_argument("--out_dir", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--src_json", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--ref_npz", required=True)
    p.add_argument("--ar_npz", required=True)
    p.add_argument("--locations", required=True)
    p.add_argument("--out_dir", required=True)
    p = sub.add_parser("viz")
    p.add_argument("--ar_npz", required=True)
    p.add_argument("--locations", required=True)
    p.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    {"locate": cmd_locate, "extract": cmd_extract, "compare": cmd_compare,
     "viz": cmd_viz}[args.cmd](args)


if __name__ == "__main__":
    main()
