#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a compact ground-truth pickle for Waymo E2E ADE/RFS scoring.

WHY THIS EXISTS
---------------
``fast_ddrive/eval/evaluate_waymo_metrics.py`` can take ``--gt`` as EITHER a
tfrecord glob OR a pre-saved ``.pkl``.  Scoring a predictions.json against the
val set therefore normally requires shipping the entire ~226 GB raw val
tfrecord tree into the (internal) scoring env just to read three float fields
per frame.  This script extracts ONLY the 479 rater-scored ("rated") frames and
the exact GT object the metric consumes, then pickles it.  The resulting pkl is
a few MB and is byte-compatible with the metric's ``--gt *.pkl`` code path.

WHAT THE METRIC EXPECTS FROM A .pkl  (derived, NOT guessed)
-----------------------------------------------------------
In ``evaluate_waymo_metrics.py:main()`` the pkl branch is::

    with open(args.gt, "rb") as f:
        gt_dict = pickle.load(f)
    ...
    for frame_name in gt_dict:
        data = gt_dict[frame_name]
        gt_traj = np.stack([data.future_states.pos_x, data.future_states.pos_y], axis=1)
        ...
        for j in range(len(data.preference_trajectories)):
            ... data.preference_trajectories[j].pos_x
            ... data.preference_trajectories[j].pos_y
            ... data.preference_trajectories[j].preference_score
        vel_x = data.past_states.vel_x[-1]
        vel_y = data.past_states.vel_y[-1]

So the pkl is ``{frame_name: data}`` where:
  * ``frame_name``  == ``E2EDFrame.frame.context.name`` (same key predictions use,
    see convert_wod_e2e.py:224 ``sid = fr.frame.context.name``).
  * ``data`` is any object exposing, by attribute:
      - ``data.future_states.pos_x``  -> 1-D sequence of float (20 @ 4 Hz)
      - ``data.future_states.pos_y``  -> 1-D sequence of float (20 @ 4 Hz)
      - ``data.past_states.vel_x``    -> 1-D sequence of float (``[-1]`` used)
      - ``data.past_states.vel_y``    -> 1-D sequence of float (``[-1]`` used)
      - ``data.preference_trajectories`` -> list; each element exposes
            ``.pos_x`` (1-D float), ``.pos_y`` (1-D float),
            ``.preference_score`` (float).

The metric NEVER re-parses a pkl; it just unpickles and reads those attributes.
``data.future_states.pos_x`` is fed to ``np.stack``/``np.asarray`` and the rater
trajectories to ``np.stack([... .pos_x, ... .pos_y], axis=-1)``, so any
sequence-of-float (we use ``np.ndarray``) works identically to the live proto
path (``load_waymo_e2e_data`` keeps the parsed proto objects).

PORTABILITY DECISION
--------------------
We do NOT pickle raw E2EDFrame protobuf objects (proto messages don't pickle
portably and would force the proto module + matching version into the scoring
env) and we do NOT pickle raw serialized bytes (convert_wod_e2e.py:245 does that
for its OWN --gt_pkl, but that produces ``{name: bytes}`` which the metric's pkl
path would NOT understand — it would try ``bytes.future_states`` and crash).
Instead each value is a ``types.SimpleNamespace`` (stdlib, present in every env)
of ``numpy.ndarray`` fields.  numpy is already a hard dependency of the metric,
so the pkl unpickles with zero extra deps.

Proto field reference (waymo_open_dataset/protos/end_to_end_driving_data.proto):
  E2EDFrame.frame.context.name        (key)
  E2EDFrame.future_states  : EgoTrajectoryStates {pos_x, pos_y, vel_x, vel_y, ...}
  E2EDFrame.past_states    : EgoTrajectoryStates {..., vel_x, vel_y, ...}
  E2EDFrame.preference_trajectories : repeated EgoTrajectoryStates
                                      {pos_x, pos_y, ..., preference_score}

"Rated" == has preference trajectories whose first preference_score != -1, the
exact predicate the metric uses to keep GT frames (load_waymo_e2e_data lines
235-237) and that convert_wod_e2e.is_rated uses.  --rated_only (default ON) keeps
only those ~479 frames so the pkl mirrors the eval subset.

Env: the ``autovla`` conda env (tensorflow + waymo-open-dataset + compiled
``end_to_end_driving_data_pb2``), same as convert_wod_e2e.py / the metric's own
GT extraction.

Usage::

    python build_rated_val_gt.py \
        --val_tfrecords '/home/kaiwen/data/fast-ddrive/waymo/val/val_*.tfrecord*' \
        --out_pkl       /home/kaiwen/data/fast-ddrive/eval/rated_val_gt.pkl \
        --rated_only
"""
import argparse
import glob
import os
import sys

# Match the env-parameterization convention applied to prep_train_jax.py /
# parquet_to_ar_with_embeds.py: model snapshot + repo root via env, with the
# current local defaults.  (Not load-bearing here, but kept for consistency so
# this script slots into the same harness as the rest of jax_ddrive/scripts.)
SNAP = os.environ.get(
    "FASTDDRIVE_SNAP",
    "/home/kaiwen/data/fast-ddrive/models/fast_ddrive_release",
)
sys.path.insert(
    0,
    os.environ.get("FASTDDRIVE_REPO", "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive"),
)

# Sentinel matching the metric / convert_wod_e2e: an unrated frame has its first
# preference trajectory's preference_score == -1.
_UNRATED_SCORE = -1


def _is_rated(fr) -> bool:
    """Mirror convert_wod_e2e.is_rated AND evaluate_waymo_metrics.load_waymo_e2e_data:
    a rated frame has >=1 preference trajectory and its first preference_score != -1."""
    pt = fr.preference_trajectories
    return len(pt) > 0 and pt[0].preference_score != _UNRATED_SCORE


def _to_gt_object(fr):
    """Build the portable GT object the metric's --gt *.pkl path consumes.

    Returns a SimpleNamespace mirroring the proto attributes the metric reads:
      .future_states.{pos_x,pos_y}
      .past_states.{vel_x,vel_y}
      .preference_trajectories[j].{pos_x,pos_y,preference_score}

    All array fields are float32 numpy arrays so downstream np.stack / np.asarray
    behaves identically to the live proto (repeated float) path.
    """
    import numpy as np
    from types import SimpleNamespace

    fs = fr.future_states
    ps = fr.past_states

    future_states = SimpleNamespace(
        pos_x=np.asarray(fs.pos_x, dtype=np.float32),
        pos_y=np.asarray(fs.pos_y, dtype=np.float32),
    )
    # The metric only reads vel_x[-1] / vel_y[-1] from past_states, but we keep
    # the full repeated arrays so the object is faithful and robust to indexing.
    past_states = SimpleNamespace(
        vel_x=np.asarray(ps.vel_x, dtype=np.float32),
        vel_y=np.asarray(ps.vel_y, dtype=np.float32),
    )

    pref = []
    for pt in fr.preference_trajectories:
        pref.append(
            SimpleNamespace(
                pos_x=np.asarray(pt.pos_x, dtype=np.float32),
                pos_y=np.asarray(pt.pos_y, dtype=np.float32),
                preference_score=float(pt.preference_score),
            )
        )

    return SimpleNamespace(
        future_states=future_states,
        past_states=past_states,
        preference_trajectories=pref,
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Extract the 479 rater-scored val frames into a compact GT pkl that "
            "fast_ddrive/eval/evaluate_waymo_metrics.py consumes via --gt <file>.pkl."
        )
    )
    ap.add_argument(
        "--val_tfrecords",
        required=True,
        help="Glob for the val tfrecord shards (e.g. "
        "'/home/kaiwen/data/fast-ddrive/waymo/val/val_*.tfrecord*').",
    )
    ap.add_argument(
        "--out_pkl",
        required=True,
        help="Output .pkl path. Pickled object is {frame_name: GT-object}.",
    )
    ap.add_argument(
        "--rated_only",
        action="store_true",
        default=True,
        help="Keep only rater-scored frames (the ~479 eval subset). Default ON; "
        "this is what the metric's GT loader itself does.",
    )
    ap.add_argument(
        "--all_frames",
        dest="rated_only",
        action="store_false",
        help="Override --rated_only and keep ALL frames (debug; produces a huge pkl).",
    )
    ap.add_argument(
        "--max_shards",
        type=int,
        default=-1,
        help="Cap number of shards read (debug / smoke test).",
    )
    args = ap.parse_args()

    # Lazy heavy imports so --help and py_compile never pull tensorflow/proto.
    import pickle
    import tensorflow as tf
    from tqdm import tqdm
    from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e

    shards = sorted(glob.glob(args.val_tfrecords))
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    if not shards:
        print(f"[gt] ERROR: no shards matched {args.val_tfrecords}")
        print("BUILD_RATED_VAL_GT: FAIL")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.out_pkl)) or "."
    os.makedirs(out_dir, exist_ok=True)

    gt_dict = {}
    n_seen = n_rated = n_dup = 0
    print(f"[gt] reading {len(shards)} shard(s); rated_only={args.rated_only}")
    for shard in tqdm(shards, desc="shards"):
        ds = tf.data.TFRecordDataset([shard], compression_type="")
        for raw in ds:
            fr = e2e.E2EDFrame()
            fr.ParseFromString(raw.numpy())
            n_seen += 1
            if args.rated_only and not _is_rated(fr):
                continue
            name = fr.frame.context.name
            if name in gt_dict:
                n_dup += 1
                continue
            gt_dict[name] = _to_gt_object(fr)
            n_rated += 1

    # Atomic write (tmp -> replace), matching the repo's resumable-writer convention.
    tmp = args.out_pkl + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(gt_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, args.out_pkl)

    size_mb = os.path.getsize(args.out_pkl) / (1024.0 * 1024.0)
    print("=" * 60)
    print("BUILD RATED VAL GT")
    print("=" * 60)
    print(f"[gt] shards read        : {len(shards)}")
    print(f"[gt] frames seen        : {n_seen}")
    print(f"[gt] rated frames kept  : {n_rated}")
    print(f"[gt] duplicate names    : {n_dup}")
    print(f"[gt] out pkl            : {args.out_pkl} ({size_mb:.2f} MB)")
    print(f"[gt] dict size          : {len(gt_dict)}")

    if args.rated_only and len(gt_dict) != 479:
        print(
            f"[gt] WARNING: expected 479 rated frames, got {len(gt_dict)}. "
            "Verify the glob covers the full val split."
        )
    print("BUILD_RATED_VAL_GT: PASS")


if __name__ == "__main__":
    main()
