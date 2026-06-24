"""OVERNIGHT PARITY — Stage A: PyTorch oracle for the MULTIMODAL SASD training step.

Standalone (does NOT edit any existing code). Runs in the `ddrive` PyTorch env.

Given a prep npz (produced by the REAL dataset-processing code jax_ddrive/eval/prep_train_jax.py),
this:
  1. Freezes EVERY input the model consumes (Codex finding #4: full determinism).
  2. Builds the doubled [2,2L] SASD batch via the SHARED pure-numpy noiser
     ddrive_jax/diffusion/noise.py:make_batch with a DETERMINISTIC fixed mask, so the JAX
     side reproduces byte-identical doubled tensors (assert input_final equality at parity).
  3. Fuses the (frozen) ViT image embeds into BOTH doubled rows at the IMAGE_TOK positions.
  4. Runs the PyTorch forward (eval, no_grad, fp32, eager attention) -> doubled-seq logits.
  5. Computes primary (section-weighted) + complementary (causal) loss via the MODEL's OWN
     loss methods.
  6. Saves an "oracle bundle": frozen inputs (Layer1) + post-fusion tensors (Layer2) + FULL
     logits (to .npy on disk) + losses + provenance.

Also runs dataset-processing cross-checks against the model's native ops:
  - model.get_rope_index(ids, thw) == prep position_ids   (validates get_rope_index_numpy port)
  - mask invariant: fixed_mask subseteq (response & ~scaffold); final = fixed | (im_end & resp)

Usage:
  python capture_oracle_sasd_mm.py --prep_npz <dir/00000.npz> --out_dir <bundle_dir>
"""
import argparse, importlib.util, json, os, sys
import numpy as np
import torch

SNAP = os.environ.get("FASTDDRIVE_SNAP",
        "/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
REPO = "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive"
NOISE_PY = f"{REPO}/ddrive_jax/diffusion/noise.py"

MASK_ID, IM_END, NULL_ID, IMAGE_TOK = 151665, 151645, 151666, 151655


def _load_noise():
    """Import the pure-numpy noiser directly by file path (avoids ddrive_jax package __init__,
    which pulls in jax). noise.py imports only numpy, so it runs in the torch env."""
    spec = importlib.util.spec_from_file_location("sasd_noise_purenp", NOISE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def deterministic_fixed_mask(ids, labels, scaffold):
    """Pre-im_end deterministic mask, already filtered to (response & ~scaffold) so it matches
    what noise.make_batch(fixed_mask=...) expects. Pattern: every other response token."""
    L = ids.shape[0]
    resp = labels != -100
    idx = np.arange(L)
    return resp & (~scaffold) & (idx % 2 == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default=None, help="bundle name; default = prep npz stem")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or os.path.splitext(os.path.basename(args.prep_npz))[0]

    noise = _load_noise()
    d = np.load(args.prep_npz)
    ids = d["input_ids"].astype(np.int64)
    labels = d["labels"].astype(np.int64)
    rbi = d["rbi"].astype(np.int64)
    turn = d["turn"].astype(np.int64)
    scaffold = d["scaffold"].astype(bool)
    weight_vec = d["weight_vec"].astype(np.float32)
    pixel_values = d["pixel_values"]
    image_grid_thw = d["image_grid_thw"].astype(np.int64)
    position_ids = d["position_ids"].astype(np.int64)        # [3, L]
    L = ids.shape[0]
    resp = labels != -100
    print(f"[A] {tag}: L={L} resp={int(resp.sum())} scaffold={int(scaffold.sum())} "
          f"n_img_tok={int((ids==IMAGE_TOK).sum())} grid={image_grid_thw.tolist()}", flush=True)

    # ---- 1) deterministic doubled batch via the SHARED pure-numpy noiser ----
    fixed_mask = deterministic_fixed_mask(ids, labels, scaffold)
    # mask invariants (Codex #7): fixed_mask must be subseteq response & ~scaffold
    assert bool((fixed_mask <= (resp & ~scaffold)).all()), "fixed_mask leaks outside response&~scaffold"
    s = {"input_ids": ids, "labels": labels, "rbi": rbi, "scaffold": scaffold,
         "weight_vec": weight_vec, "block_alpha": d["block_alpha"], "block_beta": d["block_beta"]}
    input_final, labels_final, original_labels, weights = noise.make_batch(
        s, np.random.default_rng(0), fixed_mask=fixed_mask)   # [2,2L],[2,L],[1,L],[2,L]
    num_items = noise.num_items(s)
    # row0 noisy half holds MASK at (newly-noised) UNION (pre-existing MASK padding tokens in ids).
    mask_indices = fixed_mask | ((ids == IM_END) & resp)       # what make_batch newly masks
    n_preexist = int((ids == MASK_ID).sum())                   # bd_size padding MASK tokens (labels=-100)
    final_mask_row0 = (input_final[0, :L] == MASK_ID)
    expected_row0 = mask_indices | (ids == MASK_ID)
    assert bool((final_mask_row0 == expected_row0).all()), "row0 mask != newly-masked ∪ preexisting-MASK"
    print(f"[A] doubled: input_final={input_final.shape} newly_masked(row0)={int(mask_indices.sum())} "
          f"preexisting_MASK_pad={n_preexist} num_items={num_items}", flush=True)

    # ---- 2) load PyTorch model (fp32, eager) ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.float32, trust_remote_code=True, attn_implementation="eager").eval().cuda()
    M = model.model
    mod = sys.modules[type(model).__module__]
    hybrid_block_causal_mask_multiturn = mod.hybrid_block_causal_mask_multiturn
    V = model.config.text_config.vocab_size

    # ---- dataset-processing cross-check: model.get_rope_index == prep position_ids ----
    with torch.no_grad():
        pos_model, _ = M.get_rope_index(torch.tensor(ids)[None].cuda(),
                                        torch.tensor(image_grid_thw).cuda(), None, attention_mask=None)
    pos_model = pos_model[:, 0, :].cpu().numpy()              # [3, L]
    rope_match = bool((pos_model == position_ids).all())
    rope_maxdiff = int(np.abs(pos_model - position_ids).max())
    print(f"[A] get_rope_index match prep: {rope_match} (maxdiff={rope_maxdiff})", flush=True)

    # ---- 3) ViT image embeds (frozen), doubled + scattered into both rows ----
    with torch.no_grad():
        pv = torch.tensor(pixel_values, dtype=torch.float32).cuda()
        thw = torch.tensor(image_grid_thw).cuda()
        img_feats = M.get_image_features(pv, thw)
        img_emb = (torch.cat(img_feats, 0) if isinstance(img_feats, (list, tuple)) else img_feats)
        img_emb = img_emb.to(torch.float32)                  # [N, D]
        N, Dh = img_emb.shape
        # image-token positions in the DOUBLED [2L] sequence (row0; identical across rows)
        img_pos = np.where(input_final[0] == IMAGE_TOK)[0]   # [2N]
        assert img_pos.shape[0] == 2 * N, f"img_pos {img_pos.shape[0]} != 2N {2*N}"
        ie_doubled = torch.cat([img_emb, img_emb], 0)        # [2N, D]
        emb = M.language_model.embed_tokens(torch.tensor(input_final).cuda())  # [2,2L,D]
        emb = emb.clone()
        ipos = torch.tensor(img_pos, device=emb.device)
        for r in range(2):
            emb[r, ipos, :] = ie_doubled.to(emb.dtype)
        fused_embeds = emb                                    # [2,2L,D]

    # ---- 4) doubled positions + hybrid mask + forward ----
    twoL = 2 * L
    pos_doubled = np.concatenate([position_ids, position_ids], axis=1)        # [3, 2L]
    pos_t = torch.tensor(pos_doubled)[:, None, :].repeat(1, 2, 1).cuda()      # [3, 2, 2L]
    qi = torch.arange(twoL).view(twoL, 1); ki = torch.arange(twoL).view(1, twoL)
    dense = hybrid_block_causal_mask_multiturn(
        0, 0, qi, ki, response_block_idx=torch.tensor(rbi),
        turn_idx=torch.tensor(turn), n=L).bool()                              # [2L,2L]
    add_mask = torch.where(dense, 0.0, float("-inf")).view(1, 1, twoL, twoL).cuda()

    with torch.no_grad():
        hid = M.language_model(inputs_embeds=fused_embeds,
                               position_ids=pos_t,
                               attention_mask=add_mask.expand(2, 1, twoL, twoL),
                               use_cache=False).last_hidden_state             # [2,2L,D]
        noisy_logits = model.lm_head(hid[:, :L]).float()                      # [2,L,V]
        clean_logits = model.lm_head(hid[:1, L:]).float()                     # [1,L,V]
        primary = model.compute_section_weighted_loss(
            noisy_logits, torch.tensor(labels_final).cuda(),
            torch.tensor(weights).cuda(), V, num_items_in_batch=float(num_items))
        comp = model.loss_function(logits=clean_logits, labels=torch.tensor(original_labels).cuda(),
                                   vocab_size=V, num_items_in_batch=float(num_items))
        total = primary + comp
    print(f"[A] TORCH loss primary={float(primary):.6f} complementary={float(comp):.6f} "
          f"total={float(total):.6f}", flush=True)

    # ---- 5) save bundle ----
    nl = noisy_logits.cpu().numpy(); cl = clean_logits.cpu().numpy()
    np.save(os.path.join(args.out_dir, f"{tag}.noisy_logits.npy"), nl)
    np.save(os.path.join(args.out_dir, f"{tag}.clean_logits.npy"), cl)
    np.save(os.path.join(args.out_dir, f"{tag}.fused_embeds.npy"), fused_embeds.cpu().numpy())
    np.save(os.path.join(args.out_dir, f"{tag}.img_emb.npy"), img_emb.cpu().numpy())
    bundle = os.path.join(args.out_dir, f"{tag}.bundle.npz")
    np.savez(bundle,
             # Layer1 frozen inputs
             input_ids=ids, labels=labels, rbi=rbi, turn=turn, scaffold=scaffold,
             weight_vec=weight_vec, pixel_values=pixel_values, image_grid_thw=image_grid_thw,
             position_ids=position_ids, fixed_mask=fixed_mask,
             # Layer2 post-noising / post-fusion
             input_final=input_final, labels_final=labels_final, original_labels=original_labels,
             weights=weights, dense_mask=dense.cpu().numpy(), pos_doubled=pos_doubled,
             img_pos=img_pos,
             # scalars / outputs
             num_items=np.float64(num_items), L=np.int64(L), N_img=np.int64(N),
             primary=np.float64(float(primary)), complementary=np.float64(float(comp)),
             total=np.float64(float(total)),
             # cross-checks
             rope_match=np.bool_(rope_match), rope_maxdiff=np.int64(rope_maxdiff),
             # provenance
             dtype="float32", attn_impl="eager", snapshot=os.path.basename(SNAP))
    meta = {"tag": tag, "L": int(L), "N_img": int(N), "num_items": float(num_items),
            "primary": float(primary), "complementary": float(comp), "total": float(total),
            "rope_match": rope_match, "rope_maxdiff": rope_maxdiff,
            "noisy_logits_shape": list(nl.shape), "clean_logits_shape": list(cl.shape),
            "bundle": bundle}
    json.dump(meta, open(os.path.join(args.out_dir, f"{tag}.meta.json"), "w"), indent=2)
    print(f"[A] saved bundle {bundle}")
    print(f"ORACLE_SASD_MM_DONE tag={tag}")


if __name__ == "__main__":
    main()
