#!/usr/bin/env python3
"""Pad per-sample SASD npz to a UNIFORM (L, n_blocks) so the distilled set reuses the
validated grain fast-stacking path (max_length=None) instead of the never-exercised
variable-L `_pad_sample` loader path.

Distilled labels are longer than the pseudo labels, so prep_train_jax emits variable
L (1184..1280) and n_blocks (7..9). We right-pad the token axis to max_L and the block
axis to max_N, using the pad contract documented in
`ddrive_jax/data/grain_pipeline._pad_sample`:

    input_ids   -> MASK_ID 151665      labels       -> -100        rbi  -> -1
    turn        -> -1                  scaffold     -> True        weight_vec -> 0.0
    vision_mask -> False               position_ids -> 0 (masked)  block_alpha/beta -> 1.0

Loss-zero invariant: padded token positions get labels == -100 AND weight_vec == 0.0,
so section-weighted CE / causal CE contribute exactly zero. Padded blocks are inert:
no rbi value points at them (rbi stays < real n_blocks), so their Beta(alpha,beta) is
never sampled against any token.

Usage::
    python jax_ddrive/scripts/pad_npz_uniform.py --in_dir <npz_dir> --out_dir <padded_dir> \
        [--max_L 1280] [--max_blocks N]   # omit to auto-detect the max over in_dir
"""
import argparse
import glob
import os

import numpy as np

MASK_ID = 151665

# field -> (axis to pad on the token/L axis, pad value). position_ids handled specially.
TOKEN_PAD = {
    "input_ids": MASK_ID,
    "labels": -100,
    "rbi": -1,
    "turn": -1,
    "scaffold": True,
    "weight_vec": 0.0,
    "vision_mask": False,
}
BLOCK_PAD = {"block_alpha": 1.0, "block_beta": 1.0}
PASSTHROUGH = ("pixel_values", "image_grid_thw")  # fixed shape, copied as-is


def scan_maxes(files):
    max_L = max_N = 0
    for f in files:
        d = np.load(f)
        max_L = max(max_L, int(np.asarray(d["input_ids"]).shape[0]))
        max_N = max(max_N, int(d["n_blocks"]))
    # round L up to a multiple of bd_size 32 (should already be, but be safe)
    max_L = ((max_L + 31) // 32) * 32
    return max_L, max_N


def pad_one(d, max_L, max_N):
    out = {}
    L = int(np.asarray(d["input_ids"]).shape[0])
    nb = int(d["n_blocks"])
    padL, padN = max_L - L, max_N - nb
    assert padL >= 0 and padN >= 0, f"sample L={L}/nb={nb} exceeds max_L={max_L}/max_N={max_N}"

    for k, val in TOKEN_PAD.items():
        a = np.asarray(d[k])
        out[k] = np.concatenate([a, np.full(padL, val, dtype=a.dtype)]) if padL else a
    # position_ids: (3, L) -> pad axis 1 with 0 (masked positions, value irrelevant)
    pos = np.asarray(d["position_ids"])
    out["position_ids"] = np.pad(pos, ((0, 0), (0, padL)), constant_values=0) if padL else pos
    for k, val in BLOCK_PAD.items():
        a = np.asarray(d[k])
        out[k] = np.concatenate([a, np.full(padN, val, dtype=a.dtype)]) if padN else a
    for k in PASSTHROUGH:
        out[k] = np.asarray(d[k])
    # n_blocks scalar -> max_N (uniform); inert padded blocks carry no tokens.
    out["n_blocks"] = np.asarray(max_N, dtype=np.asarray(d["n_blocks"]).dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_L", type=int, default=0, help="0 = auto-detect over in_dir")
    ap.add_argument("--max_blocks", type=int, default=0, help="0 = auto-detect over in_dir")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "*.npz")))
    assert files, f"no npz under {args.in_dir}"
    os.makedirs(args.out_dir, exist_ok=True)

    auto_L, auto_N = scan_maxes(files)
    max_L = args.max_L or auto_L
    max_N = args.max_blocks or auto_N
    assert max_L >= auto_L and max_N >= auto_N, \
        f"requested max_L={max_L}/max_N={max_N} < observed {auto_L}/{auto_N}"
    print(f"[pad] {len(files)} npz -> uniform L={max_L}, n_blocks={max_N} "
          f"(observed max L={auto_L}, blocks={auto_N})", flush=True)

    for i, f in enumerate(files):
        out = pad_one(np.load(f), max_L, max_N)
        np.savez(os.path.join(args.out_dir, os.path.basename(f)), **out)
        if (i + 1) % 1000 == 0:
            print(f"[pad] {i+1}/{len(files)}", flush=True)
    print(f"[pad] done -> {args.out_dir}  (uniform L={max_L}, n_blocks={max_N})")
    print("PAD_DONE")


if __name__ == "__main__":
    main()
