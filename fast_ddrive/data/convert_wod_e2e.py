#!/usr/bin/env python3
"""Convert Waymo Open Dataset End-to-End (WOD-E2E) tfrecords → Fast-dDrive JSON.

This is the preprocessing script the repo README promised ("will be added").  It
reads ``E2EDFrame`` protos and emits the exact JSON schema consumed by
``fast_ddrive/eval/batch_inference.py`` (and the JAX eval), plus extracts the
three front-view camera JPEGs.

The canonical Fast-dDrive prompt (reverse-engineered byte-for-byte from
``fast_ddrive/data/example/sample.json``) has a FIXED instruction prefix and
three per-frame fields:
  * ``<image>``  — placeholder; batch_inference inserts all 3 front cams here.
  * High-level navigation command — ``EgoIntent.Intent`` enum name (GO_STRAIGHT/
    GO_LEFT/GO_RIGHT; UNKNOWN→GO_STRAIGHT).
  * Historical ego state — 7 points at 0.5 s over the last 3 s, taken from
    ``past_states`` (16 pts @ 0.25 s, last = origin) at indices 3,5,7,9,11,13,15.

``sample_id`` is set to ``frame.context.name`` so predictions join the GT in
``evaluate_waymo_metrics.py`` (which keys GT on ``frame.context.name``).

Env: the ``autovla`` conda env (tensorflow + waymo-open-dataset + compiled
``end_to_end_driving_data_pb2``).  Usage::

    python fast_ddrive/data/convert_wod_e2e.py \
        --tfrecords '/data/waymo/val/val_*.tfrecord*' \
        --out_json  /data/eval/val_rated.json \
        --image_root /data/eval/val_images \
        --rated_only            # eval set = rater-scored frames only (~479)
"""
import argparse
import glob
import json
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e

# ── CameraName enum ints (confirmed from a real frame) ──────────────────────
CAM_FRONT = 1
CAM_FRONT_LEFT = 2
CAM_FRONT_RIGHT = 3
# prompt order: "left front, center front, right front"
FRONT_TRIPLET = [(CAM_FRONT_LEFT, "FRONT_LEFT"), (CAM_FRONT, "FRONT"), (CAM_FRONT_RIGHT, "FRONT_RIGHT")]

# past_states: 16 pts @ 0.25 s (idx 15 = t=0).  Prompt wants 7 pts @ 0.5 s over last 3 s.
HIST_IDX = [3, 5, 7, 9, 11, 13, 15]
HIST_LABELS = ["t-3.0s", "t-2.5s", "t-2.0s", "t-1.5s", "t-1.0s", "t-0.5s", "t+0.0s"]

# Fast-dDrive fixed instruction prefix (byte-for-byte from sample.json).
PROMPT_PREFIX = (
    "You are an expert autonomous driving agent.\n"
    "Task 1: Critical Object Detection\n"
    "For each class below, answer \"yes\" or \"no\" to indicate whether it affects the ego "
    "vehicle’s behavior or future trajectory:\n"
    "[nearby_vehicle, pedestrian, cyclist, construction, traffic_element, weather_condition, "
    "road_hazard, emergency_vehicle, animal, special_vehicle, conflicting_vehicle, "
    "door_opening_vehicle]\n"
    "Task 2: Scene Reasoning\n"
    "Predict the future behavior of the identified critical objects and explain how the "
    "identified critical objects or conditions affect the ego vehicle’s next 3-second "
    "trajectory.\n"
    "Task 3: Meta-Behavior Prediction\n"
    "Predict the ego vehicle’s future meta-driving behavior:\n"
    "- speed ∈ {keep, accelerate, decelerate, stop, other}\n"
    "- command ∈ {straight, yield, left_turn, right_turn, lane_follow, lane_change_left, "
    "lane_change_right, reverse, overtake, other}\n"
    "Task 4: Trajectory Prediction\n"
    "Predict the optimal 5-second future trajectory (5 waypoints, 1 s intervals).\n\n"
    "Input:\n"
    "- <image>: three front-view frames from left front, center front, right front cameras. \n"
    "- High-level navigation command: {NAV}\n"
    "- Historical ego state: Provided are the previous ego vehicle status recorded over the "
    "last 3.0 seconds (at 0.5-second intervals).{HIST}\n\n"
    "Output:"
)


def intent_to_nav(intent: int) -> str:
    name = e2e.EgoIntent.Intent.Name(intent) if intent in e2e.EgoIntent.Intent.values() else "UNKNOWN"
    return "GO_STRAIGHT" if name == "UNKNOWN" else name


def build_history(ps) -> str:
    """7 ego-state points at 0.5 s, formatted exactly like sample.json."""
    n = len(ps.pos_x)
    entries = []
    for idx, lab in zip(HIST_IDX, HIST_LABELS):
        i = min(idx, n - 1)
        x, y = ps.pos_x[i], ps.pos_y[i]
        ax = ps.accel_x[i] if i < len(ps.accel_x) else 0.0
        ay = ps.accel_y[i] if i < len(ps.accel_y) else 0.0
        vx = ps.vel_x[i] if i < len(ps.vel_x) else 0.0
        vy = ps.vel_y[i] if i < len(ps.vel_y) else 0.0
        entries.append(
            f"({lab}) [{x:.2f}, {y:.2f}], Acceleration: X {ax:.2f}, Y {ay:.2f} m/s², "
            f"Velocity: X {vx:.2f}, Y {vy:.2f} m/s,"
        )
    return "; ".join(entries)


def build_prompt(intent: int, ps) -> str:
    return PROMPT_PREFIX.replace("{NAV}", intent_to_nav(intent)).replace("{HIST}", build_history(ps))


def future_waypoints_20(fs):
    """20-pt @4 Hz GT (the metric's ADE target)."""
    return [[float(fs.pos_x[i]), float(fs.pos_y[i])] for i in range(len(fs.pos_x))]


def gt_5waypoints(fs):
    """5 waypoints at 1 s (indices 3,7,11,15,19 of the 4 Hz future) — the model's output grid."""
    idx = [3, 7, 11, 15, 19]
    return [[float(fs.pos_x[i]), float(fs.pos_y[i])] for i in idx if i < len(fs.pos_x)]


def _fmt(v):
    return f"{'+' if v >= 0 else '-'}{abs(v):05.2f}"


def build_meta_behavior(fr):
    """Derive future_meta_behavior from the GT trajectory + intent (raw WOD-E2E has no
    text labels). longitudinal from waypoint speeds; lateral from EgoIntent.
    NOTE (weak label): lateral is a coarse intent-only pseudo-label in {go straight,
    turn left, turn right}; it intentionally omits the canonical 'lane follow'/lane-change/
    yield vocab (not deterministically recoverable from intent). The trajectory (real GT)
    is the genuine supervised signal; do not assume canonical lateral-token parity."""
    wps = gt_5waypoints(fr.future_states)
    pts = [[0.0, 0.0]] + wps
    seg = [((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2) ** 0.5
           for i in range(1, len(pts))]
    v0 = seg[0] if seg else 0.0
    vf = seg[-1] if seg else 0.0
    if vf < 1.0:
        longitudinal = "come to stop"
    elif vf > v0 * 1.15:
        longitudinal = "speed up"
    elif vf < v0 * 0.85:
        longitudinal = "slow down"
    else:
        longitudinal = "keep speed"
    nav = intent_to_nav(fr.intent)
    lateral = {"GO_STRAIGHT": "go straight", "GO_LEFT": "turn left",
               "GO_RIGHT": "turn right"}.get(nav, "go straight")
    return longitudinal, lateral


def build_target(fr) -> str:
    """Full Fast-dDrive gpt JSON target.  trajectory = REAL GT (5 wp @1s); meta derived;
    critical_objects pseudo ('no' — no perception labels in raw WOD-E2E); explanation
    templated.  Documented weak-labeling: the trajectory is the genuine supervised signal."""
    co = {k: "no" for k in ["nearby_vehicle", "pedestrian", "cyclist", "construction",
                            "traffic_element", "weather_condition", "road_hazard",
                            "emergency_vehicle", "animal", "special_vehicle",
                            "conflicting_vehicle", "door_opening_vehicle"]}
    lon, lat = build_meta_behavior(fr)
    wps = gt_5waypoints(fr.future_states)
    traj = "[" + ", ".join("[" + _fmt(x) + "," + _fmt(y) + "]" for x, y in wps) + "]"
    explanation = (f"The ego vehicle is expected to {lon} and {lat} along its current path. "
                   "No critical objects are detected in the scene that require immediate "
                   "evasive action, so the planned trajectory follows the navigation command.")
    obj = {"critical_objects": co, "explanation": explanation,
           "future_meta_behavior": {"longitudinal": lon, "lateral": lat},
           "trajectory": traj}
    return json.dumps(obj, ensure_ascii=False)


def is_rated(fr) -> bool:
    pt = fr.preference_trajectories
    return len(pt) > 0 and pt[0].preference_score != -1


def save_front_images(fr, image_root: str, sample_id: str, rel_prefix: str):
    by_name = {im.name: im.image for im in fr.frame.images}
    rels = []
    for cam_id, tag in FRONT_TRIPLET:
        if cam_id not in by_name:
            return None  # missing a front cam → skip frame
        rel = f"{rel_prefix}/{sample_id}_{tag}.jpg"
        dst = os.path.join(image_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(by_name[cam_id])  # raw JPEG bytes, no re-encode
        rels.append(rel)
    return rels


def main():
    ap = argparse.ArgumentParser(description="WOD-E2E tfrecord → Fast-dDrive JSON converter.")
    ap.add_argument("--tfrecords", required=True, help="Glob for tfrecord shards.")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--image_root", required=True, help="Front-cam JPEGs written under here; JSON paths are relative to it.")
    ap.add_argument("--rel_prefix", default="images", help="Subdir under image_root for JPEGs.")
    ap.add_argument("--rated_only", action="store_true", help="Keep only rater-scored frames (the eval subset).")
    ap.add_argument("--with_target", action="store_true",
                    help="Populate the gpt answer with a full target (GT trajectory + derived "
                         "meta + pseudo text) for training. Raw WOD-E2E has no text labels.")
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--max_shards", type=int, default=-1)
    ap.add_argument("--gt_pkl", default="", help="Optional: also pickle the rated GT dict {name: E2EDFrame} for fast metrics.")
    args = ap.parse_args()

    shards = sorted(glob.glob(args.tfrecords))
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    assert shards, f"no shards matched {args.tfrecords}"
    os.makedirs(args.image_root, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)

    samples = []
    gt_dict = {}
    n_seen = n_kept = n_skip_img = 0
    for shard in tqdm(shards, desc="shards"):
        ds = tf.data.TFRecordDataset([shard], compression_type="")
        for raw in ds:
            fr = e2e.E2EDFrame()
            fr.ParseFromString(raw.numpy())
            n_seen += 1
            if args.rated_only and not is_rated(fr):
                continue
            sid = fr.frame.context.name
            rels = save_front_images(fr, args.image_root, sid, args.rel_prefix)
            if rels is None:
                n_skip_img += 1
                continue
            prompt = build_prompt(fr.intent, fr.past_states)
            gpt_value = build_target(fr) if args.with_target else ""
            sample = {
                "sample_id": sid,
                "image": rels,
                "navigation_command": intent_to_nav(fr.intent),
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": gpt_value},  # target (--with_target) or empty (eval)
                ],
                "future waypoints": future_waypoints_20(fr.future_states),
                "gt_5waypoints_1s": gt_5waypoints(fr.future_states),
                "rated": bool(is_rated(fr)),
            }
            samples.append(sample)
            if args.gt_pkl and is_rated(fr):
                gt_dict[sid] = raw.numpy()
            n_kept += 1
            if args.max_frames > 0 and n_kept >= args.max_frames:
                break
        if args.max_frames > 0 and n_kept >= args.max_frames:
            break

    with open(args.out_json, "w") as f:
        json.dump(samples, f)
    print(f"[convert] shards={len(shards)} seen={n_seen} kept={n_kept} skip_missing_cam={n_skip_img}")
    print(f"[convert] wrote {len(samples)} samples → {args.out_json}")
    print(f"[convert] images → {args.image_root}/{args.rel_prefix}")
    if args.gt_pkl and gt_dict:
        import pickle
        with open(args.gt_pkl, "wb") as f:
            pickle.dump(gt_dict, f)
        print(f"[convert] gt raw bytes ({len(gt_dict)}) → {args.gt_pkl}")
    print("CONVERT_DONE")


if __name__ == "__main__":
    main()
