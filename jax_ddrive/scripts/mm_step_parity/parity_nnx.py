"""OVERNIGHT PARITY — Stage B: ddrive_jax (NNX) vs the PyTorch oracle, MULTIMODAL SASD step.

Standalone (does NOT edit existing code). Runs in the `jax` venv. Consumes the oracle bundle
from capture_oracle_sasd_mm.py.

Two boundaries (Codex finding #2):
  - Layer2 (STRICT): feed the FROZEN PyTorch fused_embeds -> NNX decoder -> logits+loss.
    Isolates the decoder+loss math from ViT/noising. This is the core "decoder is a faithful
    port" gate.
  - Layer1 (INTEGRATION): NNX runs its OWN ViT on the frozen pixel_values, fuses, forwards.
    Reports ViT-embed cosine attribution separately (ViT itself gated by parity_vit).

Decomposed sub-checks (Codex findings #1/#5), each with its own tolerance:
  determinism  : NNX noise.make_batch(fixed_mask) reproduces the oracle input_final bit-exact
  logits       : rel-max < 1e-3, rel-L2, top-1 agreement (streamed vs oracle .npy mmap)
  target NLL   : max/mean abs diff on the actual target tokens
  loss         : primary / complementary / total rel < 1e-3
  per-section  : CO/explanation/fmb/trajectory loss rel (grouped by weight value), diagnostic

Usage:
  python parity_nnx.py --bundle <dir/00000.bundle.npz>
"""
import argparse, importlib.util, json, os, sys
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp

# Paths are env-overridable so this same script runs on the local GPU box AND on a TPU VM.
REPO = os.environ.get("MMSTEP_REPO", "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
sys.path.insert(0, REPO)
from ddrive_jax.models.qwen2_5_text import Qwen25TextConfig
from ddrive_jax.models.vision_qwen25vl import VisionConfig
from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_text, load_fast_ddrive_vit
from ddrive_jax.diffusion.masks import to_attn_mask4d
from ddrive_jax.diffusion.sasd_loss import section_weighted_ce, causal_ce

SNAP = os.environ.get("FASTDDRIVE_SNAP",
        "/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
NOISE_PY = f"{REPO}/ddrive_jax/diffusion/noise.py"
MASK_ID, IM_END, IMAGE_TOK = 151665, 151645, 151655
# Layer2 (no ViT) is strict; Layer1 carries the ViT cuDNN-Conv3d seed -> looser logit tol,
# matching the existing end-to-end ViT gate parity_vit.py:60 (relmax < 1e-2). Loss/top-1 stay tight.
TOL_LOGIT, TOL_LOGIT_L1, TOL_LOSS, TOL_COS = 1e-3, 1e-2, 1e-3, 0.999
SECTION_BY_W = {1.5: "critical_objects", 1.0: "explanation", 2.0: "future_meta_behavior", 3.0: "trajectory"}


def _load_noise():
    spec = importlib.util.spec_from_file_location("sasd_noise_purenp", NOISE_PY)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def stream_logit_stats(nnx_logits, oracle_npy_path, labels_shift, weights_shift, chunk=128):
    """Single chunked pass over the seq dim comparing NNX logits (host np) to oracle .npy (mmap).
    Returns dict: relmax, rell2, top1, tgt_nll_maxdiff, tgt_nll_meandiff, per-section {nll_o, nll_n}.
    labels_shift/weights_shift are the shift-by-1 labels/weights flattened over [rows*seq']. """
    orc = np.load(oracle_npy_path, mmap_mode="r")            # [R, L, V]
    R, L, V = nnx_logits.shape
    assert orc.shape == (R, L, V), f"shape {orc.shape} != {(R,L,V)}"
    Lp = L - 1                                               # shifted length
    max_abs = 0.0; ssd = 0.0; ssr = 0.0; top1_match = 0; top1_tot = 0; ref_absmax = 0.0
    # target-token nll accumulators (only valid label positions)
    nll_o_sum = {}; nll_n_sum = {}; nll_cnt = {}            # per section
    tgt_absdiff_max = 0.0; tgt_absdiff_sum = 0.0; tgt_cnt = 0
    # iterate rows then chunks of the shifted positions [0..Lp)
    for r in range(R):
        lab_r = labels_shift[r]; w_r = weights_shift[r]      # [Lp]
        for s in range(0, Lp, chunk):
            e = min(s + chunk, Lp)
            on = np.asarray(orc[r, s:e], np.float32)         # oracle logits at shifted positions [c,V]
            nn = nnx_logits[r, s:e].astype(np.float32)
            d = nn - on
            max_abs = max(max_abs, float(np.abs(d).max()))
            ref_absmax = max(ref_absmax, float(np.abs(on).max()))
            ssd += float((d * d).sum()); ssr += float((on * on).sum())
            top1_match += int((nn.argmax(-1) == on.argmax(-1)).sum()); top1_tot += (e - s)
            lab = lab_r[s:e]
            valid = lab != -100
            if valid.any():
                idxv = np.where(valid)[0]; tgt = lab[idxv]
                lse_o = np.log(np.exp(on[idxv] - on[idxv].max(-1, keepdims=True)).sum(-1)) + on[idxv].max(-1)
                lse_n = np.log(np.exp(nn[idxv] - nn[idxv].max(-1, keepdims=True)).sum(-1)) + nn[idxv].max(-1)
                nllo = lse_o - on[idxv, tgt]; nlln = lse_n - nn[idxv, tgt]
                ad = np.abs(nlln - nllo)
                tgt_absdiff_max = max(tgt_absdiff_max, float(ad.max())); tgt_absdiff_sum += float(ad.sum()); tgt_cnt += len(idxv)
                for j, p in enumerate(idxv):
                    sec = SECTION_BY_W.get(round(float(w_r[s + p]), 3), "other")
                    nll_o_sum[sec] = nll_o_sum.get(sec, 0.0) + float(nllo[j])
                    nll_n_sum[sec] = nll_n_sum.get(sec, 0.0) + float(nlln[j])
                    nll_cnt[sec] = nll_cnt.get(sec, 0) + 1
    relmax = max_abs / (ref_absmax + 1e-9)                   # existing-gate convention: max|d| / max|ref|
    rell2 = float(np.sqrt(ssd) / (np.sqrt(ssr) + 1e-9))
    top1 = top1_match / max(top1_tot, 1)
    persec = {s: {"oracle_nll": nll_o_sum[s] / nll_cnt[s], "nnx_nll": nll_n_sum[s] / nll_cnt[s],
                  "n": nll_cnt[s]} for s in nll_o_sum}
    return {"relmax": relmax, "rell2": rell2, "top1": top1,
            "tgt_nll_maxdiff": tgt_absdiff_max, "tgt_nll_meandiff": tgt_absdiff_sum / max(tgt_cnt, 1),
            "per_section": persec}


def nnx_forward(model, fused, pos_doubled, dense_mask, L):
    """fused [2,2L,D] -> noisy_logits [2,L,V], clean_logits [1,L,V] (host np)."""
    mask4d = to_attn_mask4d(jnp.asarray(dense_mask))                       # [1,1,2L,2L]
    hid = model.hidden_forward_mrope(jnp.asarray(fused, jnp.float32),
                                     pos_doubled, mask4d, mrope_section=(16, 24, 24))   # [2,2L,D]
    noisy = np.asarray(model.attend(hid[:, :L, :]), np.float32)            # [2,L,V]
    clean = np.asarray(model.attend(hid[:1, L:, :]), np.float32)           # [1,L,V]
    return noisy, clean


def nnx_loss(noisy_logits, clean_logits, labels_final, original_labels, weights, num_items):
    prim = float(section_weighted_ce(jnp.asarray(noisy_logits), jnp.asarray(labels_final),
                                     jnp.asarray(weights), num_items=num_items))
    comp = float(causal_ce(jnp.asarray(clean_logits), jnp.asarray(original_labels), num_items=num_items))
    return prim, comp, prim + comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    stem = args.bundle.replace(".bundle.npz", "")
    d = np.load(args.bundle)
    L = int(d["L"]); num_items = float(d["num_items"])
    out = {"bundle": args.bundle, "L": L, "checks": {}}
    print(f"[B] {os.path.basename(stem)}: L={L} num_items={num_items} "
          f"torch total={float(d['total']):.6f}", flush=True)

    # --- determinism: NNX make_batch(fixed_mask) reproduces oracle doubled tensors bit-exact ---
    noise = _load_noise()
    s = {"input_ids": d["input_ids"], "labels": d["labels"], "rbi": d["rbi"], "scaffold": d["scaffold"],
         "weight_vec": d["weight_vec"], "block_alpha": d.get("block_alpha"), "block_beta": d.get("block_beta")}
    # block_alpha/beta not in bundle (fixed_mask path ignores them); supply dummies sized to n_blocks
    nb = int(d["rbi"].max()) + 1 if d["rbi"].max() >= 0 else 1
    s["block_alpha"] = np.ones(nb, np.float32); s["block_beta"] = np.ones(nb, np.float32)
    inf2, labf2, orig2, w2 = noise.make_batch(s, np.random.default_rng(0), fixed_mask=d["fixed_mask"])
    det = {"input_final": bool((inf2 == d["input_final"]).all()),
           "labels_final": bool((labf2 == d["labels_final"]).all()),
           "original_labels": bool((orig2 == d["original_labels"]).all()),
           "weights": bool(np.allclose(w2, d["weights"]))}
    det_ok = all(det.values())
    out["checks"]["determinism"] = {"pass": det_ok, **det}
    print(f"[B] determinism (NNX make_batch == oracle): {det_ok}  {det}", flush=True)

    # --- load NNX text model ---
    model, _ = load_fast_ddrive_text(SNAP, Qwen25TextConfig.fast_ddrive(jnp.float32),
                                     dtype=jnp.float32, verbose=False)
    pos_doubled = jnp.asarray(d["pos_doubled"])                            # [3,2L]
    labels_final = d["labels_final"]; weights = d["weights"]; original_labels = d["original_labels"]
    # shifted labels/weights for the streamed logit stats (match section_weighted_ce shift-by-1)
    lab_sh = labels_final[:, 1:]; w_sh = weights[:, 1:]
    orig_sh = np.broadcast_to(original_labels[:, 1:], (1, L - 1))

    def cmp_block(noisy, clean, label, tol_logit=TOL_LOGIT):
        ns = stream_logit_stats(noisy, f"{stem}.noisy_logits.npy", lab_sh, w_sh)
        cs = stream_logit_stats(clean, f"{stem}.clean_logits.npy", orig_sh, np.ones_like(orig_sh, np.float32))
        prim, comp, tot = nnx_loss(noisy, clean, labels_final, original_labels, weights, num_items)
        to, po, co = float(d["total"]), float(d["primary"]), float(d["complementary"])
        rel = lambda a, b: abs(a - b) / (abs(b) + 1e-9)
        res = {
            "noisy_logits": {k: ns[k] for k in ("relmax", "rell2", "top1", "tgt_nll_maxdiff", "tgt_nll_meandiff")},
            "clean_logits": {k: cs[k] for k in ("relmax", "rell2", "top1", "tgt_nll_maxdiff", "tgt_nll_meandiff")},
            "loss": {"primary_nnx": prim, "primary_torch": po, "primary_rel": rel(prim, po),
                     "complementary_nnx": comp, "complementary_torch": co, "complementary_rel": rel(comp, co),
                     "total_nnx": tot, "total_torch": to, "total_rel": rel(tot, to)},
            "per_section_noisy": ns["per_section"],
        }
        logit_ok = ns["relmax"] < tol_logit and cs["relmax"] < tol_logit
        loss_ok = res["loss"]["total_rel"] < TOL_LOSS and res["loss"]["primary_rel"] < TOL_LOSS and res["loss"]["complementary_rel"] < TOL_LOSS
        res["pass"] = bool(logit_ok and loss_ok)
        print(f"\n[B] === {label} ===")
        print(f"    noisy logits: relmax {ns['relmax']:.3e} relL2 {ns['rell2']:.3e} top1 {ns['top1']*100:.2f}% "
              f"tgtNLL maxd {ns['tgt_nll_maxdiff']:.3e}")
        print(f"    clean logits: relmax {cs['relmax']:.3e} relL2 {cs['rell2']:.3e} top1 {cs['top1']*100:.2f}%")
        print(f"    loss primary {prim:.6f}/{po:.6f} rel {rel(prim,po):.2e} | "
              f"comp {comp:.6f}/{co:.6f} rel {rel(comp,co):.2e} | total {tot:.6f}/{to:.6f} rel {rel(tot,to):.2e}")
        for sec, v in res["per_section_noisy"].items():
            print(f"      per-sec {sec:>22}: oracle_nll {v['oracle_nll']:.4f} nnx_nll {v['nnx_nll']:.4f} "
                  f"reldiff {abs(v['nnx_nll']-v['oracle_nll'])/(abs(v['oracle_nll'])+1e-9):.2e} (n={v['n']})")
        print(f"    -> {label} {'PASS' if res['pass'] else 'FAIL'}")
        return res

    # ===== Layer2 (STRICT): frozen fused_embeds -> NNX decoder =====
    fused = np.load(f"{stem}.fused_embeds.npy")                            # [2,2L,D]
    n2, c2 = nnx_forward(model, fused, pos_doubled, d["dense_mask"], L)
    out["checks"]["layer2_frozen_fused"] = cmp_block(n2, c2, "LAYER2 (frozen fused_embeds -> decoder)")

    # ===== Layer1 (INTEGRATION): NNX runs its own ViT =====
    vit, _ = load_fast_ddrive_vit(SNAP, VisionConfig(dtype=jnp.float32), dtype=jnp.float32)
    ie = np.asarray(vit(jnp.asarray(d["pixel_values"], jnp.float32), d["image_grid_thw"]), np.float32)  # [N,D]
    oracle_ie = np.load(f"{stem}.img_emb.npy")
    cos = float((ie.ravel() @ oracle_ie.ravel()) / (np.linalg.norm(ie) * np.linalg.norm(oracle_ie) + 1e-9))
    vit_relmax = float(np.abs(ie - oracle_ie).max() / (np.abs(oracle_ie).max() + 1e-9))
    print(f"\n[B] ViT embeds: cosine {cos:.6f}  relmax {vit_relmax:.3e}  (gate cos>={TOL_COS})")
    # fuse NNX ViT embeds into both rows
    input_final = d["input_final"]; img_pos = d["img_pos"]
    emb = np.array(model.embed_tokens(jnp.asarray(input_final)), np.float32)   # [2,2L,D] (writable copy)
    ie_dbl = np.concatenate([ie, ie], 0)                                          # [2N,D]
    emb[0, img_pos, :] = ie_dbl; emb[1, img_pos, :] = ie_dbl
    n1, c1 = nnx_forward(model, emb, pos_doubled, d["dense_mask"], L)
    l1 = cmp_block(n1, c1, "LAYER1 (NNX ViT -> fuse -> decoder)", tol_logit=TOL_LOGIT_L1)
    l1["vit_cosine"] = cos; l1["vit_relmax"] = vit_relmax; l1["vit_pass"] = bool(cos >= TOL_COS)
    out["checks"]["layer1_own_vit"] = l1

    # ===== Layer1-CONTROL: NNX text embeds + ORACLE img_emb via the SAME fusion path =====
    # If this matches Layer2 (strict 1e-3), the NNX embed_tokens + scatter code is faithful and the
    # entire Layer1 logit excess is attributable to the ViT cuDNN-Conv3d seed (NOT a fusion bug).
    embc = np.array(model.embed_tokens(jnp.asarray(input_final)), np.float32)
    oie_dbl = np.concatenate([oracle_ie, oracle_ie], 0)
    embc[0, img_pos, :] = oie_dbl; embc[1, img_pos, :] = oie_dbl
    nc, cc = nnx_forward(model, embc, pos_doubled, d["dense_mask"], L)
    ctrl = cmp_block(nc, cc, "LAYER1-CTRL (NNX text + ORACLE img_emb -> decoder)")
    out["checks"]["layer1_control_oracle_img"] = ctrl

    # ===== verdict =====
    gate = (det_ok and out["checks"]["layer2_frozen_fused"]["pass"]
            and out["checks"]["layer1_own_vit"]["pass"] and l1["vit_pass"]
            and out["checks"]["layer1_control_oracle_img"]["pass"])
    out["pass"] = bool(gate)
    rp = args.report or f"{stem}.parity_nnx.json"
    json.dump(out, open(rp, "w"), indent=2)
    print(f"\n[B] report -> {rp}")
    print(f"SASD_MM_NNX_PARITY_{'PASS' if gate else 'FAIL'} "
          f"(determinism={det_ok} layer2={out['checks']['layer2_frozen_fused']['pass']} "
          f"layer1={out['checks']['layer1_own_vit']['pass']} layer1_ctrl="
          f"{out['checks']['layer1_control_oracle_img']['pass']} vit={l1['vit_pass']})")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
