#!/usr/bin/env python3
"""READ-ONLY dataset probe for Fast-dDrive / WOD-E2E data (item 2a).

One command to answer "what *is* this dataset, actually?" for a weak internal agent.
It is a PROBE, not a checker: it prints a clean human report and never asserts/raises
on schema mismatch (a missing field is reported, not fatal).

Two mutually-exclusive modes:

  --raw <tfrecord_glob>
      Parse the first N (default 5) Waymo ``E2EDFrame`` protos and print, per frame:
      presence of ``frame.context.name``, #images + their CameraName ids/names, the
      intent enum name, ``len(past_states)`` / ``len(future_states)`` poses, and whether
      ``preference_trajectories[0].preference_score`` exists (i.e. the frame is rater-
      scored). Also prints the total record count across all matched shards.
      Mirrors the proto field access + camera enum ints (1/2/3) + ``intent_to_nav`` in
      ``fast_ddrive/data/convert_wod_e2e.py``.  Needs the ``autovla`` conda env
      (tensorflow + ``waymo_open_dataset.protos.end_to_end_driving_data_pb2``).

  --processed <dir>
      Auto-detect ``*.arrayrecord`` vs ``*.parquet``, open the FIRST record, and print
      every field name + dtype + shape, the total record count, and (AR) whether
      ``image_embeds`` is present.  Reuses the canonical decoders:
        * AR      -> ``ddrive_jax.data.ar_dataset.decode_example`` / ``_ar_dtypes``
        * Parquet -> ``ddrive_jax.data.parquet_dataset.decode_row``
      Also reads & prints the ``dataset_info_*.json`` sidecar if present.

Heavy deps are imported lazily: tensorflow + waymo only under --raw; the AR/parquet
decoders only under --processed.  Importing this module (or ``--help``) pulls nothing
beyond the stdlib + numpy.

Env (matching prep_train_jax.py / parquet_to_ar_with_embeds.py convention):
  FASTDDRIVE_REPO  default /home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive
  FASTDDRIVE_SNAP  default the local Fast-dDrive HF snapshot (unused here, kept for parity)

Example:
  # processed (ArrayRecord with embeds)
  python jax_ddrive/scripts/probe_dataset.py \
      --processed /home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd_val_v2_ar
  # raw waymo tfrecords
  python jax_ddrive/scripts/probe_dataset.py \
      --raw '/home/kaiwen/data/fast-ddrive/waymo/val/val_*.tfrecord*'
"""
import argparse
import glob
import json
import os
import sys

# Repo on sys.path so ``ddrive_jax.data.*`` decoders import under --processed. Parameterized
# to match the convention applied to prep_train_jax.py / parquet_to_ar_with_embeds.py.
FASTDDRIVE_REPO = os.environ.get(
    "FASTDDRIVE_REPO", "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
FASTDDRIVE_SNAP = os.environ.get(
    "FASTDDRIVE_SNAP",
    "/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
    "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")

# CameraName enum ints (confirmed in convert_wod_e2e.py from a real frame).
CAM_NAMES = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}


def _describe(name, val):
    """Human one-liner for a decoded field: dtype + shape (numpy) or scalar repr."""
    import numpy as np
    if isinstance(val, np.ndarray):
        return f"{name:18s} dtype={str(val.dtype):10s} shape={tuple(val.shape)}"
    if isinstance(val, (bytes, str)):
        s = val.decode() if isinstance(val, bytes) else val
        return f"{name:18s} (scalar str)  value={s!r}"
    return f"{name:18s} (scalar {type(val).__name__}) value={val!r}"


# ────────────────────────────────────────────────────────────────────────────
#  --raw : Waymo E2EDFrame tfrecords
# ────────────────────────────────────────────────────────────────────────────
def probe_raw(pattern, n):
    print(f"[probe:raw] pattern = {pattern}")
    shards = sorted(glob.glob(pattern))
    if not shards:
        print(f"[probe:raw] NO SHARDS matched {pattern!r}")
        print("PROBE_DONE")
        return
    print(f"[probe:raw] matched {len(shards)} shard(s); first = {shards[0]}")

    # Lazy heavy imports (autovla env only).
    try:
        import tensorflow as tf
        from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e
    except Exception as exc:  # noqa: BLE001 - probe must never crash on import
        print(f"[probe:raw] FAILED to import tensorflow / waymo proto: {exc!r}")
        print("[probe:raw] hint: run inside the `autovla` conda env "
              "(tensorflow + waymo_open_dataset with compiled "
              "end_to_end_driving_data_pb2).")
        print("PROBE_DONE")
        return

    def intent_to_nav(intent):
        # Mirrors convert_wod_e2e.intent_to_nav.
        try:
            name = (e2e.EgoIntent.Intent.Name(intent)
                    if intent in e2e.EgoIntent.Intent.values() else "UNKNOWN")
        except Exception:  # noqa: BLE001
            name = "UNKNOWN"
        return "GO_STRAIGHT" if name == "UNKNOWN" else name

    # Total record count across all shards (cheap iteration, no proto parse).
    total = 0
    for shard in shards:
        try:
            for _ in tf.data.TFRecordDataset([shard], compression_type=""):
                total += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[probe:raw] WARN could not count {os.path.basename(shard)}: {exc!r}")
    print(f"[probe:raw] total records across {len(shards)} shard(s) = {total}")

    # Parse the first n frames from the first shard.
    print(f"[probe:raw] --- first {n} frame(s) from {os.path.basename(shards[0])} ---")
    ds = tf.data.TFRecordDataset([shards[0]], compression_type="")
    shown = 0
    for raw in ds:
        if shown >= n:
            break
        fr = e2e.E2EDFrame()
        try:
            fr.ParseFromString(raw.numpy())
        except Exception as exc:  # noqa: BLE001
            print(f"[frame {shown}] PARSE FAILED: {exc!r}")
            shown += 1
            continue

        sid = fr.frame.context.name
        has_name = bool(sid)
        imgs = list(fr.frame.images)
        cam_ids = [im.name for im in imgs]
        cam_tags = [CAM_NAMES.get(cid, f"CAM_{cid}") for cid in cam_ids]
        nav = intent_to_nav(fr.intent)
        n_past = len(fr.past_states.pos_x)
        n_future = len(fr.future_states.pos_x)
        pt = fr.preference_trajectories
        rated = len(pt) > 0 and pt[0].preference_score != -1
        score = pt[0].preference_score if len(pt) > 0 else None

        print(f"[frame {shown}]")
        print(f"  context.name present : {has_name}  value={sid!r}")
        print(f"  #images              : {len(imgs)}  cam_ids={cam_ids} "
              f"names={cam_tags}")
        print(f"  intent (raw enum)    : {fr.intent}  -> nav={nav}")
        print(f"  len(past_states)     : {n_past}  (pos_x points)")
        print(f"  len(future_states)   : {n_future}  (pos_x points)")
        print(f"  rated (pref_score)   : {rated}  "
              f"(#pref_traj={len(pt)}, score={score})")
        shown += 1

    print(f"[probe:raw] shown {shown} frame(s).")
    print("PROBE_DONE")


# ────────────────────────────────────────────────────────────────────────────
#  --processed : ArrayRecord or Parquet SASD shards
# ────────────────────────────────────────────────────────────────────────────
def _print_sidecar(data_dir):
    matches = sorted(glob.glob(os.path.join(data_dir, "dataset_info_*.json")))
    if not matches:
        print("[probe:processed] no dataset_info_*.json sidecar found.")
        return
    for path in matches:
        print(f"[probe:processed] sidecar {os.path.basename(path)}:")
        try:
            info = json.load(open(path))
        except Exception as exc:  # noqa: BLE001
            print(f"  (could not read: {exc!r})")
            continue
        for key in ("split", "num_samples", "num_shards", "shard_size",
                    "byte_order", "shape_suffix"):
            if key in info:
                print(f"  {key:12s}: {info[key]}")
        if "array_dtypes" in info:
            print(f"  array_dtypes: {list(info['array_dtypes'].keys())}")
        files = info.get("files")
        if files:
            print(f"  files       : {len(files)} (e.g. {files[0]})")


def _probe_arrayrecord(data_dir, ar_files):
    print(f"[probe:processed] format = ArrayRecord ({len(ar_files)} shard file(s))")
    # Lazy import of the canonical decoder (pulls TF only via decode_example).
    sys.path.insert(0, FASTDDRIVE_REPO)
    from ddrive_jax.data.ar_dataset import decode_example, _ar_dtypes

    try:
        from array_record.python.array_record_data_source import ArrayRecordDataSource
    except Exception as exc:  # noqa: BLE001
        print(f"[probe:processed] FAILED to import array_record: {exc!r}")
        print("PROBE_DONE")
        return
    src = ArrayRecordDataSource(ar_files)
    total = len(src)
    print(f"[probe:processed] total records (all shards) = {total}")
    if total == 0:
        print("[probe:processed] empty source.")
        print("PROBE_DONE")
        return

    sample = decode_example(src[0])
    # _ar_dtypes is keyed by the TF parse module; report the *expected* AR field set.
    try:
        import tensorflow as tf
        expected = list(_ar_dtypes(tf).keys())
        print(f"[probe:processed] expected AR array fields: {expected}")
    except Exception:  # noqa: BLE001
        pass

    print("[probe:processed] --- first record fields ---")
    for k in sorted(sample.keys()):
        print("  " + _describe(k, sample[k]))
    has_embeds = "image_embeds" in sample
    print(f"[probe:processed] image_embeds present : {has_embeds} "
          f"({'dataset v2' if has_embeds else 'v1 / pixels-only'})")
    _print_sidecar(data_dir)
    print("PROBE_DONE")


def _probe_parquet(data_dir, pq_files):
    print(f"[probe:processed] format = Parquet ({len(pq_files)} shard file(s))")
    sys.path.insert(0, FASTDDRIVE_REPO)
    from ddrive_jax.data.parquet_dataset import decode_row

    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # noqa: BLE001
        print(f"[probe:processed] FAILED to import pyarrow: {exc!r}")
        print("PROBE_DONE")
        return

    total = 0
    for path in pq_files:
        try:
            total += pq.read_metadata(path).num_rows
        except Exception as exc:  # noqa: BLE001
            print(f"[probe:processed] WARN could not read metadata "
                  f"{os.path.basename(path)}: {exc!r}")
    print(f"[probe:processed] total records (all shards) = {total}")

    rows = pq.read_table(pq_files[0]).to_pylist()
    if not rows:
        print("[probe:processed] first shard has 0 rows.")
        print("PROBE_DONE")
        return
    sample = decode_row(rows[0])
    print("[probe:processed] --- first record fields ---")
    for k in sorted(sample.keys()):
        print("  " + _describe(k, sample[k]))
    has_embeds = "image_embeds" in sample
    print(f"[probe:processed] image_embeds present : {has_embeds} "
          "(parquet shards are typically pixels-only / v1)")
    _print_sidecar(data_dir)
    print("PROBE_DONE")


def probe_processed(data_dir):
    print(f"[probe:processed] dir = {data_dir}")
    if not os.path.isdir(data_dir):
        print(f"[probe:processed] NOT A DIRECTORY: {data_dir}")
        print("PROBE_DONE")
        return
    ar_files = sorted(glob.glob(os.path.join(data_dir, "*.arrayrecord")))
    pq_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if ar_files:
        _probe_arrayrecord(data_dir, ar_files)
    elif pq_files:
        _probe_parquet(data_dir, pq_files)
    else:
        print(f"[probe:processed] no *.arrayrecord or *.parquet under {data_dir}")
        _print_sidecar(data_dir)
        print("PROBE_DONE")


def main():
    ap = argparse.ArgumentParser(
        description="READ-ONLY probe of Fast-dDrive raw (Waymo tfrecord) or "
                    "processed (ArrayRecord/Parquet) datasets. Prints a human report; "
                    "never asserts on schema.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--raw", metavar="TFRECORD_GLOB",
                   help="Glob for Waymo E2EDFrame tfrecord shards (autovla env).")
    g.add_argument("--processed", metavar="DIR",
                   help="Dir of *.arrayrecord or *.parquet SASD shards.")
    ap.add_argument("-n", "--num", type=int, default=5,
                    help="(--raw) number of frames to dump in detail (default 5).")
    args = ap.parse_args()

    if args.raw:
        probe_raw(args.raw, args.num)
    else:
        probe_processed(args.processed)


if __name__ == "__main__":
    main()
