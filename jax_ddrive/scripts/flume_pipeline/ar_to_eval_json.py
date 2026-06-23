"""Adapter: flume_pipeline msgpack ArrayRecord -> Fast-dDrive eval JSON (+ images on disk).

Lets the INFERENCE path consume the SAME new AR as training (doc: "reuse the two examples").
Unpacks each record (prompt_text, 3 JPEG blobs, optional future_xy) and writes the eval-JSON
schema `prep_jax_eval.py` expects: {sample_id, image:[3 rel paths], conversations:[{from:human,
value:prompt}], "future waypoints":[[x,y]...]}. Pure-python (msgpack + PIL); runs in any env.

Run:
    python ar_to_eval_json.py --ar_path <new.array_record> \
        --out_json <eval.json> --image_root <dir>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grain_loader import read_raw_records          # noqa: E402
from etl_record import unpack                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar_path", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--image_root", required=True, help="JPEGs written under here; JSON paths relative to it")
    ap.add_argument("--rel_prefix", default="images")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.image_root, args.rel_prefix), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    tags = ("FRONT_LEFT", "FRONT", "FRONT_RIGHT")

    samples = []
    for blob in read_raw_records(args.ar_path):
        rec = unpack(blob)
        sid = rec["sample_id"]
        rels = []
        for img_bytes, tag in zip(rec["images"], tags):
            rel = f"{args.rel_prefix}/{sid}_{tag}.jpg"
            with open(os.path.join(args.image_root, rel), "wb") as f:
                f.write(img_bytes)                              # raw JPEG, no re-encode
            rels.append(rel)
        fw = []
        if "future_xy" in rec:
            fw = np.frombuffer(rec["future_xy"], np.float32).reshape(-1, 2).tolist()
        samples.append({
            "sample_id": sid,
            "image": rels,
            "conversations": [{"from": "human", "value": rec["prompt_text"]},
                              {"from": "gpt", "value": ""}],   # empty target for eval
            "future waypoints": fw,
        })
    json.dump(samples, open(args.out_json, "w"))
    print(f"[ar->eval-json] {len(samples)} samples -> {args.out_json}; images -> {args.image_root}/{args.rel_prefix}")
    print(f"[ar->eval-json] sample_ids={[s['sample_id'] for s in samples]}")
    print("AR_TO_EVAL_JSON_DONE")


if __name__ == "__main__":
    main()
