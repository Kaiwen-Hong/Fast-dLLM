"""SASD parquet -> ArrayRecord(tf.train.Example) WITH precomputed frozen-ViT image embeds.

Dataset v2 (decision 2026-06-12): every record keeps ALL 12 existing array fields —
including pixel_values, so embeds stay re-verifiable from stored pixels — and adds
``image_embeds`` = frozen Fast-dDrive ViT output, computed fp32 on GPU and stored
bfloat16, as a SINGLE copy ``[N_img_tokens=672/4=168, 2048]``.  The training loader
doubles it via ``concat([ie, ie], 0)`` — mirroring
``maxtext.diffusion.sasd.compute_fast_ddrive_image_embeds``, which also runs the ViT
per sample (K=1), so this offline pass has no batching divergence from the online path.

Layout: 1:1 source-parquet-file -> AR shard, atomic .tmp->os.replace, resumable
(skip existing shards).  In-run verification every --verify_every records:
  (a) jitted recompute must be BITWISE equal to the stored fp32 result, and
  (b) tf serialize->parse roundtrip of the bf16 tensor must be byte-exact.

Run (jax venv, GPU; TF pinned to CPU in-process):
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$REPO/jax_ddrive \
  python parquet_to_ar_with_embeds.py <SRC_PARQUET_DIR> <DST_AR_DIR> [--verify_every 256]
"""
import argparse
import gc
import glob
import json
import os
import sys
import time

import numpy as np

# TF strictly CPU: JAX owns the GPU in this process. Hide before any TF op runs.
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import jax
jax.config.update("jax_default_matmul_precision", "highest")   # true fp32 (no TF32) — parity convention
import jax.numpy as jnp
import ml_dtypes
import pyarrow.parquet as pq
from array_record.python.array_record_module import ArrayRecordWriter
from flax import nnx

sys.path.insert(0, os.environ.get("FASTDDRIVE_REPO", "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive"))
from ddrive_jax.convert.hf_to_jax import _set
from ddrive_jax.data.parquet_dataset import decode_row
from ddrive_jax.models.vision_qwen25vl import (VisionConfig, VisionTransformer,
                                               _seg_ids_from_cu, cu_seqlens_full,
                                               get_window_index)

SNAP_DEFAULT = os.environ.get("FASTDDRIVE_SNAP",
                "/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
                "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")

ARRAY_FIELDS = ["input_ids", "labels", "rbi", "turn", "scaffold", "weight_vec",
                "block_alpha", "block_beta", "position_ids", "vision_mask",
                "pixel_values", "image_grid_thw"]
DTYPES = {"input_ids": "int64", "labels": "int64", "rbi": "int32", "turn": "int32",
          "scaffold": "bool", "weight_vec": "float32", "block_alpha": "float32",
          "block_beta": "float32", "position_ids": "int32", "vision_mask": "bool",
          "pixel_values": "float16", "image_grid_thw": "int64",
          "image_embeds": "bfloat16"}


def _st_tensor_f32(path, key):
    """Read one safetensors tensor as fp32 numpy, tolerant of BF16/F16/F32.
    safetensors' numpy framework cannot decode bf16, so parse the header and
    decode raw bytes via ml_dtypes. Bit-identical to
    ``f.get_tensor(key).astype(np.float32)`` for F32/F16 sources (the release
    snapshot is F32); the BASE Qwen snapshot is BF16, hence this path."""
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        hdr = json.loads(fh.read(n))
        data_start = 8 + n
        m = hdr[key]
        b, e = m["data_offsets"]
        fh.seek(data_start + b)
        raw = fh.read(e - b)
    npdt = {"F32": np.float32, "F16": np.float16, "BF16": ml_dtypes.bfloat16}[m["dtype"]]
    return np.frombuffer(raw, dtype=npdt).reshape(m["shape"]).astype(np.float32)


def load_vit_streaming(snapshot_dir):
    """Stream-load the frozen ViT one tensor at a time (~1 GB host peak, never the full
    state dict).  Mirror of the proven maxtext waymo_sasd_data_processing loader."""
    from safetensors import safe_open

    cfg = VisionConfig(dtype=jnp.float32)
    vit = VisionTransformer(cfg, rngs=nnx.Rngs(0))

    want = {"visual.patch_embed.proj.weight": ("patch_embed/kernel", "conv")}
    for i in range(cfg.depth):
        p, q = f"visual.blocks.{i}.", f"blocks/{i}/"
        want[p + "norm1.weight"] = (q + "norm1/weight", None)
        want[p + "norm2.weight"] = (q + "norm2/weight", None)
        want[p + "attn.qkv.weight"] = (q + "attn/qkv/kernel", "T")
        want[p + "attn.qkv.bias"] = (q + "attn/qkv/bias", None)
        want[p + "attn.proj.weight"] = (q + "attn/proj/kernel", "T")
        want[p + "attn.proj.bias"] = (q + "attn/proj/bias", None)
        for m in ("gate_proj", "up_proj", "down_proj"):
            want[p + f"mlp.{m}.weight"] = (q + f"mlp/{m}/kernel", "T")
            want[p + f"mlp.{m}.bias"] = (q + f"mlp/{m}/bias", None)
    want["visual.merger.ln_q.weight"] = ("merger/ln_q/weight", None)
    want["visual.merger.mlp.0.weight"] = ("merger/fc1/kernel", "T")
    want["visual.merger.mlp.0.bias"] = ("merger/fc1/bias", None)
    want["visual.merger.mlp.2.weight"] = ("merger/fc2/kernel", "T")
    want["visual.merger.mlp.2.bias"] = ("merger/fc2/bias", None)

    n = 0
    for fname in sorted(os.listdir(snapshot_dir)):
        if not fname.endswith(".safetensors"):
            continue
        sf_path = os.path.join(snapshot_dir, fname)
        with safe_open(sf_path, framework="numpy") as f:
            keys = list(f.keys())   # listing is bf16-safe; decoding is not
        for k in keys:
            if k not in want:
                continue
            path, tfm = want[k]
            arr = _st_tensor_f32(sf_path, k)   # BF16/F16/F32 -> fp32 (base ckpt is bf16)
            if tfm == "T":
                arr = arr.T
            elif tfm == "conv":
                arr = arr.reshape(arr.shape[0], -1).T
            _set(vit, path, jnp.asarray(arr))
            del arr
            n += 1
        gc.collect()
    assert n == 1 + 12 * cfg.depth + 5, f"unexpected ViT tensor count {n}"
    return vit, n


def _bytes(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _int64(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))


def to_example(r, embeds_bf16):
    feat = {k: _bytes(tf.io.serialize_tensor(tf.constant(r[k])).numpy()) for k in ARRAY_FIELDS}
    feat["image_embeds"] = _bytes(tf.io.serialize_tensor(tf.constant(embeds_bf16)).numpy())
    feat["sample_id"] = _bytes(str(r["sample_id"]).encode())
    feat["L"] = _int64(r["L"])
    feat["n_blocks"] = _int64(r["n_blocks"])
    return tf.train.Example(features=tf.train.Features(feature=feat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--snap", default=SNAP_DEFAULT)
    ap.add_argument("--split", default="train")
    ap.add_argument("--verify_every", type=int, default=256)
    ap.add_argument("--max_files", type=int, default=0, help="0=all (smoke: limit)")
    args = ap.parse_args()

    pfiles = sorted(glob.glob(os.path.join(args.src, f"{args.split}-*.parquet")))
    assert pfiles, f"no {args.split}-*.parquet under {args.src}"
    if args.max_files > 0:
        pfiles = pfiles[: args.max_files]
    nsh = len(pfiles)
    os.makedirs(args.dst, exist_ok=True)

    vit, ntens = load_vit_streaming(args.snap)
    print(f"[embeds] frozen ViT loaded ({ntens} tensors, fp32) on {jax.devices()[0]}", flush=True)

    # The ViT __call__ mixes host-side numpy (window index, rotary, seg masks — all pure
    # functions of grid_thw, which is constant across this dataset) with device compute, so
    # it can't be jitted as-is.  We precompute the grid-derived constants ONCE per distinct
    # grid and jit only the device part (a faithful mirror of VisionTransformer.__call__,
    # cross-checked against the validated eager path on every shard's first record).
    def make_jit_fwd(grid_i64):
        cfg = vit.cfg
        grid = np.asarray(grid_i64)
        window_index, cu_win = get_window_index(grid, cfg)
        cu_full = cu_seqlens_full(grid)
        cos, sin = vit._rotary(grid, window_index)            # eager: consts
        N = int(np.prod(grid, axis=1).sum())
        u = cfg.spatial_merge_unit
        seg_full = jnp.asarray(_seg_ids_from_cu(cu_full, N))
        mask_full = seg_full[:, None] == seg_full[None, :]
        seg_win = jnp.asarray(_seg_ids_from_cu(cu_win, N))
        mask_win = seg_win[:, None] == seg_win[None, :]
        wi = jnp.asarray(window_index)
        rev = jnp.asarray(np.argsort(window_index))
        fullatt = set(cfg.fullatt_block_indexes)

        @nnx.jit
        def fwd(m, pixel):                                    # pixel [N,1176] f32
            x = m.patch_embed(pixel)
            x = x.reshape(N // u, u, -1)[wi].reshape(N, -1)
            for i, blk in enumerate(m.blocks):
                x = blk(x, cos, sin, mask_full if i in fullatt else mask_win)
            x = m.merger(x)
            return x[rev]

        return fwd

    fwd_cache = {}

    def embeds_fp32(pixel_f16, grid_i64):
        key = grid_i64.tobytes()
        if key not in fwd_cache:
            fwd_cache[key] = make_jit_fwd(grid_i64)
        out = fwd_cache[key](vit, jnp.asarray(pixel_f16, jnp.float32))
        return np.asarray(out)  # [N//4, 2048] fp32

    t0, n_total, n_verified, emb_shape = time.time(), 0, 0, None
    eager_diffs = []
    for i, pf in enumerate(pfiles):
        out = os.path.join(args.dst, f"{args.split}-{i:05d}-of-{nsh:05d}.arrayrecord")
        src_rows = pq.read_metadata(pf).num_rows
        if os.path.exists(out):
            print(f"[{i+1}/{nsh}] skip (exists): {os.path.basename(out)}", flush=True)
            n_total += src_rows
            continue
        tmp = out + ".tmp"
        w = ArrayRecordWriter(tmp, "group_size:1")
        n_shard, ts = 0, time.time()
        for batch in pq.ParquetFile(pf).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                r = decode_row(row)
                e32 = embeds_fp32(r["pixel_values"], r["image_grid_thw"])
                e16 = e32.astype(ml_dtypes.bfloat16)
                emb_shape = e16.shape
                if n_shard == 0:
                    # cross-check the jitted mirror against the VALIDATED eager __call__.
                    # Two-tier criterion: a per-sample hard cap at 1e-3 catches structural
                    # bugs (wrong mask/rope -> O(0.1)); a rolling MEDIAN cap at 1e-4 catches
                    # systematic precision drift (TF32 -> ~7.5e-4 on EVERY sample) while
                    # tolerating rare fp32 reduction-noise tails (observed up to ~2.5e-4 on
                    # isolated samples; bf16 storage resolution is ~4e-3, so tails are
                    # immaterial to what is stored).
                    ref = np.asarray(vit(jnp.asarray(r["pixel_values"], jnp.float32),
                                         r["image_grid_thw"]))
                    d = float(np.max(np.abs(e32 - ref)) / (np.max(np.abs(ref)) + 1e-12))
                    eager_diffs.append(d)
                    assert d < 1e-3, f"jit mirror diverges from eager ViT: rel {d:.2e}"
                    if len(eager_diffs) >= 8:
                        med = float(np.median(eager_diffs))
                        assert med < 1e-4, (
                            f"systematic jit-vs-eager elevation: median {med:.2e} over "
                            f"{len(eager_diffs)} shards (TF32 regression?)")
                if n_total % args.verify_every == 0:
                    e32b = embeds_fp32(r["pixel_values"], r["image_grid_thw"])
                    assert np.array_equal(e32, e32b), f"non-deterministic recompute @ {r['sample_id']}"
                    ser = tf.io.serialize_tensor(tf.constant(e16)).numpy()
                    back = tf.io.parse_tensor(ser, tf.bfloat16).numpy()
                    assert back.tobytes() == e16.tobytes(), f"bf16 roundtrip mismatch @ {r['sample_id']}"
                    n_verified += 1
                w.write(to_example(r, e16).SerializeToString())
                n_shard += 1
                n_total += 1
        w.close()
        assert n_shard == src_rows, f"shard {i}: wrote {n_shard} != source {src_rows}"
        os.replace(tmp, out)
        rate = n_shard / max(time.time() - ts, 1e-9)
        eta_min = (sum(pq.read_metadata(p).num_rows for p in pfiles[i+1:]) / max(rate, 1e-9)) / 60 \
            if i + 1 < nsh else 0.0
        print(f"[{i+1}/{nsh}] {os.path.basename(out)}: {n_shard} rec, {rate:.1f}/s, "
              f"eta {eta_min:.0f}min, verified {n_verified}", flush=True)

    info = {"format": "arrayrecord/tf.train.Example", "split": args.split,
            "num_samples": n_total, "num_shards": nsh,
            "record_encoding": "tf.train.Example; array fields = tf.io.serialize_tensor bytes; "
                               "decode via tf.io.parse_tensor(bytes, dtype)",
            "array_fields": ARRAY_FIELDS + ["image_embeds"], "array_dtypes": DTYPES,
            "scalar_fields": {"sample_id": "bytes", "L": "int64", "n_blocks": "int64"},
            "image_embeds": {
                "shape_single": list(emb_shape) if emb_shape else None,
                "provenance": ("frozen ViT from snapshot below (see 'snapshot' / 'vit_source'), "
                               "fp32 forward on GPU with jax_default_matmul_precision=highest "
                               "(TF32 disabled), cast bfloat16; per-sample K=1 calls (no batching), "
                               "jitted mirror of VisionTransformer.__call__ cross-checked vs eager "
                               "per shard"),
                "vit_source": ("base-Qwen2.5-VL" if "Qwen2.5-VL" in args.snap
                               else "release-Fast-dDrive" if "Fast-dDrive" in args.snap
                               else "unknown"),
                "snapshot": args.snap,
                "doubling": "loader must concat([ie, ie], axis=0) -> [2N, D] for the doubled "
                            "[noisy|clean] sequence (first N rows = noisy half, second N = clean)",
            },
            "files": [f"{args.split}-{i:05d}-of-{nsh:05d}.arrayrecord" for i in range(nsh)]}
    json.dump(info, open(os.path.join(args.dst, f"dataset_info_{args.split}.json"), "w"), indent=2)
    # provenance/integrity manifest so a run can pin/verify which data it consumed (data_manifest.py;
    # docs/2implementation-details/DATASET_V2.md). Best-effort — never fail the build over it.
    try:
        import sys as _sys
        _sd = os.path.dirname(os.path.abspath(__file__))
        if _sd not in _sys.path:
            _sys.path.insert(0, _sd)
        from data_manifest import build_local_manifest, _git_sha
        _man = build_local_manifest(args.dst, builder_git_sha=_git_sha(_sd))
        json.dump(_man, open(os.path.join(args.dst, "DATA_MANIFEST.json"), "w"), indent=2)
        print(f"DATA_MANIFEST.json digest={_man['digest'][:16]}... "
              f"({_man['hash_kind']}, {_man['num_files']} files)", flush=True)
    except Exception as _e:
        print(f"[warn] DATA_MANIFEST not written: {_e}", flush=True)
    dmax = max(eager_diffs) if eager_diffs else float("nan")
    dmed = float(np.median(eager_diffs)) if eager_diffs else float("nan")
    print(f"AR_WITH_EMBEDS_DONE total={n_total} shards={nsh} grids={len(fwd_cache)} "
          f"emb_shape={emb_shape} verified={n_verified} "
          f"eager_rel_diff(max={dmax:.2e},median={dmed:.2e}) "
          f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
