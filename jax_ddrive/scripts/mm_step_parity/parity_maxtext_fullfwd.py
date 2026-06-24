"""OVERNIGHT PARITY — Stage C2 (STRETCH): MaxText REAL-3B FULL FORWARD vs PyTorch oracle.

Closes Codex finding #1 (transitive-closure gap): runs the ACTUAL MaxText decoder with the
REAL converted Fast-dDrive weights on the multimodal doubled SASD step, and compares the
doubled-sequence logits + loss to the PyTorch oracle. This exercises MaxText-only wiring the
math arm cannot: HF->MaxText weight mapping (param_mapping), the real-shape attention/qkv
layout, M-RoPE threading via attention_metadata, the in-attention mask interception, and the
frozen-ViT image-embed scatter in decoders._apply_embedding.

To isolate the DECODER+weights (analogue of NNX Layer2), we feed the ORACLE image embeds
through MaxText's frozen scatter path (sasd_image_embeds), so the only thing under test is the
MaxText forward, not the ViT.

Standalone (does NOT edit existing code). Run on CPU (3B x 2L=3712 fp32 won't fit one GPU):
  unset LD_LIBRARY_PATH
  BOTH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive:/home/kaiwen/Desktop/research/Fast-dLLM/maxtext-dlm-fork/src
  PYTHONPATH=$BOTH JAX_PLATFORMS=cpu /home/kaiwen/jax-dlm-baseline/.venv/bin/python parity_maxtext_fullfwd.py --bundle <...>
"""
import argparse, json, os, sys
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.models.models import transformer_as_linen
from maxtext.utils import maxtext_utils
from maxtext.diffusion.load_fast_ddrive_maxtext import build_maxtext_params_from_fast_ddrive
from maxtext.diffusion import sasd as mxt_sasd

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
HEAD_DIM, ROPE_THETA, MROPE = 128, 1_000_000.0, (16, 24, 24)
TOL_LOGIT, TOL_LOSS, TOL_TOP1 = 1e-2, 1e-3, 99.0   # decoder re-port + wide-logit noise -> 1e-2 logit, tight loss


def make_config(twoL):
    base = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
    overrides = dict(
        run_name="sasd_mm_fullfwd_parity", decoder_block="qwen2", enable_checkpointing=False,
        scan_layers=False, attention="dot_product", use_mrope=False,
        logits_via_embedding=True, logits_dot_in_fp32=True, cast_logits_to_fp32=True,
        float32_logits=True, float32_qk_product=True, matmul_precision="highest",
        normalize_embedding_logits=False, use_qk_norm=False, attention_bias=True,
        dtype="float32", weight_dtype="float32", per_device_batch_size=1.0,
        max_target_length=twoL + 8, base_emb_dim=2048, base_num_query_heads=16,
        base_num_kv_heads=2, base_mlp_dim=11008, base_num_decoder_layers=36,
        head_dim=HEAD_DIM, vocab_size=151936, mlp_activations=["silu", "linear"],
        normalization_layer_epsilon=1e-6, rope_max_timescale=ROPE_THETA, enable_dropout=False,
    )
    return pyconfig.initialize([sys.argv[0], base], override_model_config=True, **overrides)


def chunked_logit_stats(mxt_slice, oracle_npy, labels_shift, chunk=128):
    """mxt_slice [R,Lp+?,V] host np vs oracle .npy mmap [R,L,V]; compare on shifted positions."""
    orc = np.load(oracle_npy, mmap_mode="r")
    R, L, V = orc.shape
    Lp = L - 1
    max_abs = ref_absmax = ssd = ssr = 0.0; t1m = t1t = 0; nll_md = nll_sd = 0.0; nc = 0
    for r in range(R):
        lab = labels_shift[r]
        for s in range(0, Lp, chunk):
            e = min(s + chunk, Lp)
            on = np.asarray(orc[r, s:e], np.float32)
            nn_ = mxt_slice[r, s:e].astype(np.float32)
            dd = nn_ - on
            max_abs = max(max_abs, float(np.abs(dd).max())); ref_absmax = max(ref_absmax, float(np.abs(on).max()))
            ssd += float((dd * dd).sum()); ssr += float((on * on).sum())
            t1m += int((nn_.argmax(-1) == on.argmax(-1)).sum()); t1t += (e - s)
            lv = lab[s:e]; v = lv != -100
            if v.any():
                iv = np.where(v)[0]; tg = lv[iv]
                lso = np.log(np.exp(on[iv] - on[iv].max(-1, keepdims=True)).sum(-1)) + on[iv].max(-1)
                lsn = np.log(np.exp(nn_[iv] - nn_[iv].max(-1, keepdims=True)).sum(-1)) + nn_[iv].max(-1)
                ad = np.abs((lsn - nn_[iv, tg]) - (lso - on[iv, tg]))
                nll_md = max(nll_md, float(ad.max())); nll_sd += float(ad.sum()); nc += len(iv)
    return {"relmax": max_abs / (ref_absmax + 1e-9), "rell2": float(np.sqrt(ssd) / (np.sqrt(ssr) + 1e-9)),
            "top1": t1m / max(t1t, 1), "tgt_nll_maxdiff": nll_md, "tgt_nll_meandiff": nll_sd / max(nc, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    stem = args.bundle.replace(".bundle.npz", "")
    d = np.load(args.bundle)
    L = int(d["L"]); twoL = 2 * L; num_items = float(d["num_items"])
    print(f"[C2] {os.path.basename(stem)}: L={L} 2L={twoL} num_items={num_items} torch total={float(d['total']):.6f}", flush=True)

    # --- build the B=1 batch dict + prepped inputs (with ORACLE image embeds) ---
    oracle_ie = np.load(f"{stem}.img_emb.npy")                      # [N, D]
    ie_doubled = np.concatenate([oracle_ie, oracle_ie], 0)[None]    # [1, 2N, D]
    batch = {
        "input_final": d["input_final"][None], "labels_final": d["labels_final"][None],
        "original_labels": d["original_labels"][None], "weights": d["weights"][None],
        "position_ids": d["position_ids"][None], "rbi": d["rbi"][None], "turn": d["turn"][None],
        "num_items": np.asarray([num_items], np.float32),
    }
    prepped = mxt_sasd.prepare_sasd_inputs(batch, HEAD_DIM, MROPE, ROPE_THETA, image_embeds=ie_doubled)
    print(f"[C2] prepped inputs {prepped['inputs'].shape} cos {prepped['cos'].shape} "
          f"mask {prepped['attn_mask'].shape} img_embeds {prepped['image_embeds'].shape}", flush=True)

    # --- build MaxText 3B + load REAL weights ---
    cfg = make_config(twoL)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = transformer_as_linen(cfg, mesh, quant=None)
    params = build_maxtext_params_from_fast_ddrive(SNAP, cfg, verbose=True)

    inputs = prepped["inputs"]                                       # [2, 2L]
    dummy_pos = jnp.broadcast_to(jnp.arange(twoL, dtype=jnp.int32)[None], (inputs.shape[0], twoL))
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        logits, _ = model.apply(
            params, inputs, dummy_pos, decoder_segment_ids=None, enable_dropout=False,
            rngs={"dropout": jax.random.PRNGKey(0)}, mutable=["intermediates"],
            attention_metadata={
                "sasd_rope_cos": prepped["cos"], "sasd_rope_sin": prepped["sin"],
                "sasd_attn_mask": prepped["attn_mask"],
                "sasd_image_embeds": prepped["image_embeds"], "sasd_image_pos": prepped["img_pos"],
            },
        )
    logits = jnp.asarray(logits, jnp.float32)                        # [2, 2L, V]
    V = logits.shape[-1]

    # --- loss via the REAL production fn on the FULL mxt logits ---
    loss, sec_sum, cau_sum, tw = mxt_sasd.sasd_loss_from_logits(
        logits, jnp.asarray(d["labels_final"])[None], jnp.asarray(d["original_labels"])[None],
        jnp.asarray(d["weights"])[None], jnp.asarray([num_items], jnp.float32), B=1, L=L)
    mt_total = float(loss); mt_primary = float(sec_sum) / num_items; mt_comp = float(cau_sum) / num_items

    # --- logit slices to host, compare to oracle ---
    noisy = np.asarray(logits[:, :L, :])                            # [2, L, V]
    clean = np.asarray(logits[:1, L:, :])                           # [1, L, V]
    del logits
    lab_sh = d["labels_final"][:, 1:]
    orig_sh = np.broadcast_to(d["original_labels"][:, 1:], (1, L - 1))
    ns = chunked_logit_stats(noisy, f"{stem}.noisy_logits.npy", lab_sh)
    cs = chunked_logit_stats(clean, f"{stem}.clean_logits.npy", orig_sh)

    to, po, co = float(d["total"]), float(d["primary"]), float(d["complementary"])
    rel = lambda a, b: abs(a - b) / (abs(b) + 1e-9)
    logit_ok = ns["relmax"] < TOL_LOGIT and cs["relmax"] < TOL_LOGIT
    top1_ok = ns["top1"] * 100 >= TOL_TOP1 and cs["top1"] * 100 >= TOL_TOP1
    loss_ok = rel(mt_total, to) < TOL_LOSS and rel(mt_primary, po) < TOL_LOSS and rel(mt_comp, co) < TOL_LOSS
    gate = bool(logit_ok and top1_ok and loss_ok)

    print(f"\n[C2] === MaxText REAL-3B FULL FORWARD (multimodal doubled SASD) ===")
    print(f"     noisy logits: relmax {ns['relmax']:.3e} relL2 {ns['rell2']:.3e} top1 {ns['top1']*100:.2f}% tgtNLL maxd {ns['tgt_nll_maxdiff']:.3e}")
    print(f"     clean logits: relmax {cs['relmax']:.3e} relL2 {cs['rell2']:.3e} top1 {cs['top1']*100:.2f}%")
    print(f"     loss primary {mt_primary:.6f}/{po:.6f} rel {rel(mt_primary,po):.2e} | "
          f"comp {mt_comp:.6f}/{co:.6f} rel {rel(mt_comp,co):.2e} | total {mt_total:.6f}/{to:.6f} rel {rel(mt_total,to):.2e}")
    out = {"bundle": args.bundle, "L": L,
           "noisy_logits": ns, "clean_logits": cs,
           "loss": {"primary_maxtext": mt_primary, "primary_torch": po, "primary_rel": rel(mt_primary, po),
                    "complementary_maxtext": mt_comp, "complementary_torch": co, "complementary_rel": rel(mt_comp, co),
                    "total_maxtext": mt_total, "total_torch": to, "total_rel": rel(mt_total, to)},
           "logit_ok": logit_ok, "top1_ok": top1_ok, "loss_ok": loss_ok, "pass": gate}
    rp = args.report or f"{stem}.parity_maxtext_fullfwd.json"
    json.dump(out, open(rp, "w"), indent=2)
    print(f"\n[C2] report -> {rp}")
    print(f"SASD_MM_MAXTEXT_FULLFWD_PARITY_{'PASS' if gate else 'FAIL'} "
          f"(logit={logit_ok} top1={top1_ok} loss={loss_ok})")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
