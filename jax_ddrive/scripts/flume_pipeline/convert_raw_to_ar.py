"""ONE-SCRIPT raw WOD-E2E tfrecord -> msgpack ArrayRecord (the local realization of Job ①).

This is the deliverable "one script that converts [raw wode2e] to AR format". It runs in the
**autovla** conda env (tensorflow + waymo-open-dataset + the locally-compiled
`end_to_end_driving_data_pb2`), decodes raw `E2EDFrame` protos, and writes one schema-v1
msgpack record per frame via `etl_record.assemble_record` — NO JSON / NO JPEG-on-disk
intermediate (collapses the legacy convert->JSON->... chain).

It reuses `fast_ddrive/data/convert_wod_e2e.py`'s VALIDATED per-frame logic
(`build_prompt`/`build_target`/`FRONT_TRIPLET`/`is_rated`/`future_waypoints_20`) verbatim, so
the rendered prompt/target are byte-identical to the Fast-dDrive JSON the oracle consumed.
This is the local twin of the google3 `etl_flume.py` proto front-end: same record core, same
on-disk bytes; only the proto source/runner differ.

Run (autovla env):
    /home/kaiwen/miniconda3/envs/autovla/bin/python convert_raw_to_ar.py \
        --tfrecords '/home/kaiwen/data/flume_pipeline/raw_examples/two_examples.tfrecord' \
        --out /home/kaiwen/data/flume_pipeline/wod_e2e_sasd_raw.array_record
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                                   # etl_record
sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive/data")  # convert_wod_e2e

import tensorflow as tf                                                     # noqa: E402  (autovla)
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e    # noqa: E402
import convert_wod_e2e as cw                                                # noqa: E402  (validated logic)
from etl_record import assemble_record, pack                               # noqa: E402


def frame_to_record(fr, with_target=True):
    """Raw `E2EDFrame` -> schema-v1 record dict, or None to drop (missing front triplet —
    matches convert_wod_e2e.save_front_images). Prompt/target reuse cw.* so the strings are
    byte-identical to the Fast-dDrive JSON path."""
    sid = fr.frame.context.name
    by_name = {im.name: im.image for im in fr.frame.images}                # CameraName -> JPEG bytes
    if any(cam not in by_name for cam, _ in cw.FRONT_TRIPLET):
        return None
    images = [by_name[cam] for cam, _ in cw.FRONT_TRIPLET]                  # (FL, F, FR) JPEG bytes
    prompt_text = cw.build_prompt(fr.intent, fr.past_states)
    target_text = cw.build_target(fr) if with_target else ""
    ps = fr.past_states
    ego = {k: np.asarray(getattr(ps, k), np.float32)
           for k in ("pos_x", "pos_y", "accel_x", "accel_y", "vel_x", "vel_y")}
    future_xy = np.asarray(cw.future_waypoints_20(fr.future_states), np.float32)
    return assemble_record(sid, images, prompt_text, target_text, "pseudo",
                           ego=ego, intent=int(fr.intent),
                           future_xy=future_xy, rated=cw.is_rated(fr))


def main():
    ap = argparse.ArgumentParser(description="raw WOD-E2E tfrecord -> msgpack ArrayRecord")
    ap.add_argument("--tfrecords", required=True, help="glob for tfrecord shard(s)")
    ap.add_argument("--out", required=True, help="output .array_record path")
    ap.add_argument("--with_target", action="store_true", default=True,
                    help="render the gpt training target (default on; raw WOD-E2E has no text labels)")
    ap.add_argument("--no_target", dest="with_target", action="store_false")
    ap.add_argument("--max_frames", type=int, default=-1)
    args = ap.parse_args()

    from array_record.python.array_record_module import ArrayRecordWriter
    shards = sorted(glob.glob(args.tfrecords))
    assert shards, f"no shards matched {args.tfrecords}"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    writer = ArrayRecordWriter(args.out, "group_size:1")
    n_seen = n_kept = n_skip = 0
    manifest = []
    for shard in shards:
        for raw in tf.data.TFRecordDataset([shard], compression_type=""):
            fr = e2e.E2EDFrame()
            fr.ParseFromString(raw.numpy())
            n_seen += 1
            rec = frame_to_record(fr, with_target=args.with_target)
            if rec is None:
                n_skip += 1
                continue
            writer.write(pack(rec))
            n_kept += 1
            manifest.append({"sample_id": rec["sample_id"], "provenance": rec["provenance"]})
            if 0 < args.max_frames <= n_kept:
                break
        if 0 < args.max_frames <= n_kept:
            break
    writer.close()
    json.dump(manifest, open(args.out + ".manifest.json", "w"), indent=2)
    print(f"[raw->ar] seen={n_seen} kept={n_kept} skip_missing_cam={n_skip} -> {args.out}")
    print(f"[raw->ar] sample_ids={[m['sample_id'] for m in manifest]}")
    print("CONVERT_RAW_TO_AR_DONE")


if __name__ == "__main__":
    main()
