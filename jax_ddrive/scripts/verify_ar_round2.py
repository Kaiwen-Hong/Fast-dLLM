"""Round-2 semantic verification of the SASD ArrayRecord **v2** training dataset.

Verifies, for sampled records of `wod_e2e_sasd_full_v2_ar` (train) and
`wod_e2e_sasd_val_v2_ar` (val), that the stored row faithfully encodes
[image1, image2, image3, prompt, answer] **and** that the precomputed `image_embeds`
match a from-pixels recompute:

  raw tfrecord ──convert_wod_e2e.py──▶ JSON+JPEG ──prep_train_jax.py──▶ ref npz
                                                     (the EXACT build pipeline, re-run)
  v2 AR record ──ar_dataset.decode_example──▶ ar npz (12 arrays + image_embeds bf16)
  compare:  12 array columns bit-exact (ar vs ref)
          + image_embeds: frozen-ViT fp32(highest) recompute from stored pixels → bf16,
            two-tier criterion (the bf16 analogue of the v2 builder's locked fp32
            standard — stored = bf16(jit_fp32), recomputed = bf16(eager_fp32), and
            |jit−eager| ≤ δ implies |stored−recomputed| ≤ 1 ulp(x) + δ per element):
              (1) bitwise fraction ≥ 98.5% (audit observed ~99.7%),
              (2) every mismatched element is rounding-consistent: ≤1.5× its own bf16 ulp
                  OR |Δ| ≤ 2e-3 (small-magnitude elements sit below fp32 jit-vs-eager
                  noise, so their ulp distance is meaningless — bound them absolutely).
            The bf16-level global rel is REPORTED but not gated: one benign 1-ulp flip
            at a near-max element already yields ~2^-8·x/x_max > 1e-3 (the builder's
            1e-3 cap applies to fp32 pre-quantization values, not bf16 — gating on it
            here is a category error; the 06 build log's "tail sample → two-tier
            redesign" recorded the same lesson).
            Any real corruption (wrong row/image/garbage) fails (1) catastrophically.
          + structure self-checks + answer round-trip + trajectory == GT@1s
          + pixel reconstruction PNGs + embeds PCA triptych (orig | recon | PCA-RGB).

Subcommands (run under different envs; see verify_ar_round2.sh):
  locate         (pyarrow)             sample_ids -> (split, shard, row) via source parquet
  extract        (tf + array_record)   random-access read v2 AR records -> npz dumps
  embeds-verify  (jax venv, GPU)       recompute embeds from stored pixels, write checks
  compare        (ddrive env)          all checks + review.md + triptych per sample
  viz            (autovla, matplotlib) BEV / layout / M-RoPE / mask / embeds-diff figures
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
HF = "/home/kaiwen/data/fast-ddrive/hf"
sys.path.insert(0, REPO + "/jax_ddrive")

# the 12 v1 arrays compared bit-exact against the from-raw re-encoded reference
ARRAY_DTYPES = {
    "input_ids": "int64", "labels": "int64", "rbi": "int32", "turn": "int32",
    "scaffold": "bool", "weight_vec": "float32", "block_alpha": "float32",
    "block_beta": "float32", "position_ids": "int32", "vision_mask": "bool",
    "pixel_values": "float16", "image_grid_thw": "int64",
}
DATASETS = {
    "train": {"ar": f"{HF}/wod_e2e_sasd_full_v2_ar", "pq": f"{HF}/wod_e2e_sasd_full_packed",
              "prefix": "train"},
    "val": {"ar": f"{HF}/wod_e2e_sasd_val_v2_ar", "pq": f"{HF}/wod_e2e_sasd_val_pq",
            "prefix": "val"},
}
IM_START, IM_END = 151644, 151645
VSTART, VEND, VPAD, IMAGE_PAD = 151652, 151653, 151654, 151655
MASK_ID, NULL_ID = 151665, 151666
# (alpha, beta) -> section (bijective; mirrors prep_train_jax NOISE_SCHED)
AB_TO_SECTION = {(1.0, 2.0): "critical_objects", (1.0, 1.0): "explanation",
                 (1.0, 1.5): "future_meta_behavior", (2.0, 1.0): "trajectory"}
SECTION_W = {"critical_objects": 1.5, "explanation": 1.0,
             "future_meta_behavior": 2.0, "trajectory": 3.0}
# embeds acceptance (see module docstring; the bf16 analogue of the builder's standard)
EMB_MIN_BITWISE_FRAC = 0.985
EMB_ULP_RATIO_MAX = 1.5     # per-mismatch: ≤1.5× its own bf16 ulp ...
EMB_ATOL = 2e-3             # ... OR absolutely tiny (sub-fp32-noise magnitudes)


def _sid_fn(sid):
    return sid.replace(os.sep, "_")


# ──────────────────────────────────────────────────────────────── locate ────
def cmd_locate(args):
    import pyarrow.parquet as pq
    ids = json.load(open(args.ids_json))          # ordered [{"sid":..., "split":...}]
    found = {}
    for split in sorted({e["split"] for e in ids}):
        want = {e["sid"] for e in ids if e["split"] == split}
        cfg = DATASETS[split]
        files = sorted(glob.glob(os.path.join(cfg["pq"], f"{cfg['prefix']}-*.parquet")))
        assert files, f"no parquet under {cfg['pq']}"
        for si, f in enumerate(files):
            col = pq.read_table(f, columns=["sample_id"]).column("sample_id").to_pylist()
            for ri, sid in enumerate(col):
                if sid in want and sid not in found:
                    found[sid] = {"split": split, "shard": si, "row": ri,
                                  "file": os.path.basename(f)}
            if all(s in found for s in want):
                break
    order = [e["sid"] for e in ids]
    missing = [s for s in order if s not in found]
    json.dump({"order": order,
               "splits": {e["sid"]: e["split"] for e in ids},
               "locations": found, "missing": missing},
              open(args.out, "w"), indent=2)
    print(f"[locate] found {len(found)}/{len(order)}; missing={missing}")
    assert not missing, "some sample_ids not present in the source parquet"


# ─────────────────────────────────────────────────────────────── extract ────
def cmd_extract(args):
    from array_record.python.array_record_data_source import ArrayRecordDataSource
    from ddrive_jax.data.ar_dataset import decode_example
    loc = json.load(open(args.locations))
    os.makedirs(args.out_dir, exist_ok=True)
    for sid in loc["order"]:
        e = loc["locations"][sid]
        cfg = DATASETS[e["split"]]
        ar_files = sorted(glob.glob(os.path.join(cfg["ar"], f"{cfg['prefix']}-*.arrayrecord")))
        path = ar_files[e["shard"]]
        assert os.path.basename(path).startswith(f"{cfg['prefix']}-{e['shard']:05d}-"), path
        rec = decode_example(ArrayRecordDataSource([path])[e["row"]])
        assert rec["sample_id"] == sid, (
            f"ar row order mismatch: {e} has sample_id {rec['sample_id']!r}, expected {sid!r}")
        assert "image_embeds" in rec, f"{sid}: v2 record lacks image_embeds — wrong dataset?"
        emb = rec["image_embeds"]                       # ml_dtypes bfloat16 [168, 2048]
        arrs = {k: rec[k] for k in ARRAY_DTYPES}
        arrs["L"] = np.int64(rec["L"])
        arrs["n_blocks"] = np.int64(rec["n_blocks"])
        arrs["image_embeds_f32"] = emb.astype(np.float32)
        arrs["image_embeds_bits"] = emb.view(np.uint16)  # exact stored bf16 bit pattern
        np.savez(os.path.join(args.out_dir, _sid_fn(sid) + ".npz"), **arrs)
        print(f"[extract] {sid} <- {e['split']} ar shard {e['shard']} row {e['row']} "
              f"(L={rec['L']}, pixels={rec['pixel_values'].shape}, embeds={emb.shape})")
    print("EXTRACT_DONE")


# ───────────────────────────────────────────────────────── embeds-verify ────
def cmd_embeds_verify(args):
    """Recompute image_embeds from the STORED pixel_values with the validated eager ViT
    (fp32, matmul precision highest — the oracle the v2 builder was cross-checked against),
    cast bf16, and compare bit patterns against the stored embeds."""
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp
    from ddrive_jax.models.vision_qwen25vl import VisionConfig
    from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_vit

    loc = json.load(open(args.locations))
    os.makedirs(args.out_dir, exist_ok=True)
    print("[embeds] loading frozen ViT (fp32, highest) ...", flush=True)
    vit, _ = load_fast_ddrive_vit(SNAP, VisionConfig(dtype=jnp.float32),
                                  dtype=jnp.float32, verbose=False)
    n_pass = 0
    for sid in loc["order"]:
        d = np.load(os.path.join(args.ar_npz, _sid_fn(sid) + ".npz"))
        out = np.asarray(vit(jnp.asarray(d["pixel_values"], jnp.float32),
                             np.asarray(d["image_grid_thw"])), np.float32)   # [168, 2048]
        mine = np.asarray(jnp.asarray(out).astype(jnp.bfloat16))             # bf16
        mine_bits = mine.view(np.uint16)
        st_bits = d["image_embeds_bits"]
        st_f32 = d["image_embeds_f32"]
        mismatch = st_bits != mine_bits
        frac = 1.0 - float(mismatch.mean())
        mine_f32 = mine.astype(np.float32)
        ad = np.abs(st_f32 - mine_f32)[mismatch]
        mx = np.maximum(np.abs(st_f32), np.abs(mine_f32))[mismatch]
        # one bf16 ulp at each element's own magnitude (8 significand bits → exp-7)
        ulp_own = np.exp2(np.floor(np.log2(np.maximum(mx, 1e-30))) - 7)
        ratio = ad / ulp_own
        n_bad = int(((ratio > EMB_ULP_RATIO_MAX) & (ad > EMB_ATOL)).sum())
        max_abs = float(ad.max()) if ad.size else 0.0
        global_rel = max_abs / max(float(np.abs(st_f32).max()), 1e-9)  # diagnostic only
        ok = (frac >= EMB_MIN_BITWISE_FRAC) and (n_bad == 0)
        n_pass += ok
        per_tok = mismatch.sum(axis=1)                                       # [168]
        json.dump({"sample_id": sid, "bitwise_frac": frac,
                   "max_abs_diff": max_abs, "global_rel": global_rel,
                   "n_mismatch": int(mismatch.sum()), "n_over_criterion": n_bad,
                   "criterion": (f"bitwise≥{EMB_MIN_BITWISE_FRAC:.1%} AND every mismatch "
                                 f"≤{EMB_ULP_RATIO_MAX}×own-ulp or |Δ|≤{EMB_ATOL:g} "
                                 f"(bf16-level global rel reported, not gated)"),
                   "verdict": bool(ok),
                   "per_token_mismatch": per_tok.astype(int).tolist()},
                  open(os.path.join(args.out_dir, _sid_fn(sid) + ".json"), "w"))
        print(f"[embeds] {sid}: bitwise {frac:.4%}, max|Δ| {max_abs:.2e}, "
              f"global_rel {global_rel:.2e}, n_over {n_bad} -> "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    print(f"EMBEDS_VERIFY_{'PASS' if n_pass == len(loc['order']) else 'FAIL'} "
          f"({n_pass}/{len(loc['order'])})")
    sys.exit(0 if n_pass == len(loc["order"]) else 1)


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


def _pca_rgb(emb_img, gh=8, gw=7):
    """[56, 2048] f32 -> PIL image of the top-3-PCA RGB merged-token grid (gh×gw)."""
    from PIL import Image
    x = emb_img - emb_img.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    comp = x @ vt[:3].T                                  # [56, 3]
    lo, hi = np.percentile(comp, 2, axis=0), np.percentile(comp, 98, axis=0)
    rgb = np.clip((comp - lo) / np.maximum(hi - lo, 1e-9), 0, 1)
    return Image.fromarray((rgb.reshape(gh, gw, 3) * 255).astype(np.uint8))


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
        split = loc["splits"][sid]
        ar = dict(np.load(os.path.join(args.ar_npz, _sid_fn(sid) + ".npz")))
        ref = dict(np.load(os.path.join(args.ref_npz, ref_man[sid])))
        item = src_by_id[sid]
        sdir = os.path.join(args.out_dir, f"{k:02d}_{sid}")
        os.makedirs(sdir, exist_ok=True)
        checks = {}

        # 1) column-exact: ar vs freshly re-encoded reference (the 12 v1 arrays)
        for col in ARRAY_DTYPES:
            checks[f"col:{col}"] = bool(np.array_equal(ar[col], ref[col]))
        checks["col:L"] = int(ar["L"]) == int(ref["input_ids"].shape[0])
        checks["col:n_blocks"] = int(ar["n_blocks"]) == int(ref["n_blocks"])
        pv_diff = float(np.abs(ar["pixel_values"].astype(np.float32)
                               - ref["pixel_values"].astype(np.float32)).max()) \
            if ar["pixel_values"].shape == ref["pixel_values"].shape else float("nan")

        # 1b) image_embeds (v2 column): shape/dtype here; numeric verdict from embeds-verify
        emb = json.load(open(os.path.join(args.embeds_check, _sid_fn(sid) + ".json")))
        checks["col:image_embeds_shape(168,2048)bf16"] = (
            ar["image_embeds_f32"].shape == (168, 2048)
            and ar["image_embeds_bits"].dtype == np.uint16)
        checks["embeds:recompute_bf16(two-tier)"] = bool(emb["verdict"])

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
        hist = [[float(t), float(x), float(y)] for t, x, y in re_mod.findall(
            r'\(t([+-][\d.]+)s\) \[([-\d.]+), ([-\d.]+)\]',
            item["conversations"][0]["value"])]
        checks["prompt:7_history_points"] = (len(hist) == 7)

        # 4) image reconstruction + embeds PCA triptych (orig | recon | PCA)
        recon = _inverse_patchify(ar["pixel_values"], grid, mean, std)
        tags = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"]
        w, h = recon[0].size
        trip = Image.new("RGB", (3 * w + 24, 3 * h + 24), "white")
        for i, im in enumerate(recon):
            im.save(os.path.join(sdir, f"recon_{i}_{tags[i]}.png"))
            orig = Image.open(os.path.join(args.image_root, item["image"][i])).convert("RGB")
            pca = _pca_rgb(ar["image_embeds_f32"][i * 56:(i + 1) * 56]).resize(
                (w, h), Image.NEAREST)
            y = i * (h + 8)
            trip.paste(orig.resize((w, h), Image.BICUBIC), (0, y))
            trip.paste(im, (w + 12, y))
            trip.paste(pca, (2 * w + 24, y))
        trip.save(os.path.join(sdir, "triptych__orig_recon_embedsPCA.png"))
        # keep the v1-style side-by-side too (orig | recon)
        sheet = Image.new("RGB", (2 * w + 12, 3 * h + 24), "white")
        for i, im in enumerate(recon):
            orig = Image.open(os.path.join(args.image_root, item["image"][i])).convert("RGB")
            sheet.paste(orig.resize((w, h), Image.BICUBIC), (0, i * (h + 8)))
            sheet.paste(im, (w + 12, i * (h + 8)))
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
            "sample_id": sid, "split": split, "L": L, "rs": rs, "re": re_,
            "nav": item.get("navigation_command", "?"),
            "fmb_longitudinal": str(fmb.get("longitudinal", "")).replace("<|NULL|>", ""),
            "fmb_lateral": str(fmb.get("lateral", "")).replace("<|NULL|>", ""),
            "history_t_x_y": hist,
            "encoded_5wp": enc_wp,
            "gt_5wp_1s": gt5,
            "gt_20wp_4hz": item.get("future waypoints", []),
            "block_to_section": {str(b): sec for b, sec, *_ in sec_rows},
            "embeds": {"bitwise_frac": emb["bitwise_frac"],
                       "global_rel": emb["global_rel"],
                       "max_abs_diff": emb["max_abs_diff"],
                       "n_mismatch": emb["n_mismatch"], "verdict": emb["verdict"]},
        }, open(os.path.join(sdir, "viz_data.json"), "w"), indent=2)

        ok_all = all(checks.values())
        summary.append((sid, split, ok_all, checks, pv_diff, emb))

        # ── per-sample review.md ──
        with open(os.path.join(sdir, "review.md"), "w") as f:
            f.write(f"# sample {k:02d} — `{sid}` ({split})  →  "
                    f"{'✅ ALL PASS' if ok_all else '❌ FAILURES'}\n\n")
            f.write(f"- ar location: **{split}** v2 AR shard {loc['locations'][sid]['shard']}, "
                    f"row {loc['locations'][sid]['row']} (`{loc['locations'][sid]['file']}`)\n")
            f.write(f"- L = {L} (pad to %32), n_blocks = {int(ar['n_blocks'])}, "
                    f"prompt len = {rs - 3}, answer span = [{rs}, {re_}), "
                    f"MASK tail = {int((ids[re_:] == MASK_ID).sum())}\n")
            f.write(f"- image_grid_thw = {grid.tolist()}  → image_pad runs "
                    f"{[ln for _, ln in pad_runs]} (merged tokens/img), "
                    f"pixel_values {tuple(ar['pixel_values'].shape)} fp16, "
                    f"image_embeds (168, 2048) bf16\n")
            f.write(f"- max|ar.pixel_values − re-encoded.pixel_values| = {pv_diff:g}\n")
            f.write(f"- embeds recompute: bitwise {emb['bitwise_frac']:.4%}, "
                    f"max|Δ| {emb['max_abs_diff']:.2e}, global rel "
                    f"{emb['global_rel']:.2e} ({emb['criterion']})\n\n")
            f.write("## visual review files (this directory)\n\n"
                    "| file | shows |\n|---|---|\n"
                    "| `triptych__orig_recon_embedsPCA.png` | original | reconstructed-from-`pixel_values` | top-3-PCA RGB of `image_embeds` (8×7 merged-token grid) |\n"
                    "| `side_by_side__orig_vs_recon.png` | original JPEGs vs pixel reconstruction |\n"
                    "| `embeds_recompute_diff.png` | per-token bf16 mismatch counts, stored vs from-pixels recompute |\n"
                    "| `bev_trajectory.png` | BEV: ego history, encoded 5-waypoint answer, GT 20-pt future |\n"
                    "| `sequence_layout.png` | what each of the L token positions is |\n"
                    "| `position_ids.png` | 3D M-RoPE position channels |\n"
                    "| `attn_mask.png` | training hybrid block-causal attention mask |\n\n")
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
        print(f"[compare] {k:02d} {sid} ({split}): {'PASS' if ok_all else 'FAIL'} "
              f"(pv_diff={pv_diff:g}, emb_bitwise={emb['bitwise_frac']:.4%})")

    # ── top-level report ──
    n_pass = sum(1 for _, _, ok, _, _, _ in summary if ok)
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("# Round-2 semantic verification — dataset **v2** ArrayRecord "
                "(train full + val)\n\n")
        f.write(f"**{n_pass}/{len(summary)} samples pass every check.**\n\n")
        f.write("Pipeline re-run for these samples: raw tfrecord → `convert_wod_e2e.py "
                "--with_target` → `prep_train_jax.py` (the exact build chain), then the 12 "
                "array columns compared bit-exact against the v2 ArrayRecord rows, plus "
                "`image_embeds` re-derived from the stored pixels (frozen ViT, fp32 highest "
                "→ bf16) under the two-tier criterion (bitwise fraction + ≤1 ulp).\n\n")
        f.write("| # | sample_id | split | result | emb bitwise | failed checks | max px diff |\n"
                "|---|---|---|---|---|---|---|\n")
        for i, (sid, split, ok, checks, pvd, emb) in enumerate(summary):
            bad = [n for n, v in checks.items() if not v]
            f.write(f"| {i:02d} | `{sid}` | {split} | {'✅' if ok else '❌'} | "
                    f"{emb['bitwise_frac']:.3%} | {', '.join(bad) if bad else '—'} | {pvd:g} |\n")
        f.write("\nPer-sample packets (review.md + figures) are in the numbered "
                "subdirectories.\n")
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
        ar = dict(np.load(os.path.join(args.ar_npz, _sid_fn(sid) + ".npz")))
        ids, rbi, turn = ar["input_ids"], ar["rbi"], ar["turn"]
        L, rs, re_ = vd["L"], vd["rs"], vd["re"]
        b2s = {int(b): s for b, s in vd["block_to_section"].items()}

        # ── 1) BEV trajectory ──
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
        axx.set_title(f"{sid} ({vd['split']})\nnav={vd['nav']}  fmb: "
                      f"{vd['fmb_longitudinal']} / {vd['fmb_lateral']}")
        axx.axis("equal")
        axx.grid(alpha=0.3)
        axx.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(sdir, "bev_trajectory.png"), dpi=120)
        plt.close(fig)

        # ── 2) sequence layout strips ──
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

        # ── 4) training attention mask ──
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

        # ── 5) embeds recompute-diff heatmap (v2) ──
        ec = json.load(open(os.path.join(args.embeds_check, _sid_fn(sid) + ".json")))
        per_tok = np.array(ec["per_token_mismatch"]).reshape(3, 8, 7)
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
        vmax = max(int(per_tok.max()), 1)
        for i, (axx, tag) in enumerate(zip(axes, ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"])):
            im = axx.imshow(per_tok[i], cmap="Reds", vmin=0, vmax=vmax,
                            interpolation="nearest")
            axx.set_title(tag, fontsize=9)
            axx.set_xticks([]); axx.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.8, label="# mismatched bf16 elements / 2048")
        fig.suptitle(f"image_embeds: stored vs from-pixels recompute — bitwise "
                     f"{ec['bitwise_frac']:.4%}, max|Δ| {ec['max_abs_diff']:.1e}, "
                     f"global rel {ec['global_rel']:.1e} → "
                     f"{'PASS' if ec['verdict'] else 'FAIL'}", fontsize=10)
        fig.savefig(os.path.join(sdir, "embeds_recompute_diff.png"), dpi=120,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] {k:02d} {sid}: bev/layout/positions/mask/embeds-diff written")
    print("VIZ_DONE")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("locate")
    p.add_argument("--ids_json", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--locations", required=True)
    p.add_argument("--out_dir", required=True)
    p = sub.add_parser("embeds-verify")
    p.add_argument("--locations", required=True)
    p.add_argument("--ar_npz", required=True)
    p.add_argument("--out_dir", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--src_json", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--ref_npz", required=True)
    p.add_argument("--ar_npz", required=True)
    p.add_argument("--embeds_check", required=True)
    p.add_argument("--locations", required=True)
    p.add_argument("--out_dir", required=True)
    p = sub.add_parser("viz")
    p.add_argument("--ar_npz", required=True)
    p.add_argument("--embeds_check", required=True)
    p.add_argument("--locations", required=True)
    p.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    {"locate": cmd_locate, "extract": cmd_extract, "embeds-verify": cmd_embeds_verify,
     "compare": cmd_compare, "viz": cmd_viz}[args.cmd](args)


if __name__ == "__main__":
    main()
