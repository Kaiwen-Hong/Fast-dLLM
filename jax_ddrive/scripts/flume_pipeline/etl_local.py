"""Local stand-in for Job ① (the google3 Flume ETL) — doc 09 §4.3.

Reads a Fast-dDrive training JSON (the `convert_wod_e2e.py` output; default = the 2-sample
`fast_ddrive/data/example/sample.json`), builds one schema-v1 msgpack record per item via
`etl_record.build_record_from_json`, and writes them to an ArrayRecord file. This exercises
the *exact same* record core + msgpack encoding + ArrayRecord sink that the real Flume job
uses; only the front-end differs (local JSON here vs. raw `E2EDFrame` proto in `etl_flume.py`,
which is unrunnable outside google3 because the proto is `//third_party`-internal).

Run (ddrive env, `unset LD_LIBRARY_PATH`):
    python etl_local.py \
        --train_json /home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive/data/example/sample.json \
        --image_root /home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive \
        --out /home/kaiwen/data/flume_pipeline/wod_e2e_sasd.array_record
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from etl_record import build_record_from_json, pack, unpack   # noqa: E402


def write_array_record(records_bytes, out_path):
    """Write a list of msgpack-packed records to an ArrayRecord file (group_size:1 = one
    logical record per group, so Grain/`read([i])` maps 1:1 to a sample)."""
    from array_record.python.array_record_module import ArrayRecordWriter
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = ArrayRecordWriter(out_path, "group_size:1")
    for blob in records_bytes:
        writer.write(blob)
    writer.close()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--image_root", required=True,
                    help="dir that item['image'] paths are relative to (Fast-dDrive repo root)")
    ap.add_argument("--out", required=True, help="output .array_record path")
    ap.add_argument("--distilled_json", default=None,
                    help="optional {sample_id: target_text} JSON to upgrade pseudo targets")
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()

    data = json.load(open(args.train_json))
    if args.max_samples > 0:
        data = data[: args.max_samples]
    distilled = json.load(open(args.distilled_json)) if args.distilled_json else None

    records_bytes, manifest = [], []
    for k, item in enumerate(data):
        rec = build_record_from_json(item, args.image_root, distilled=distilled)
        blob = pack(rec)
        # sanity: round-trip every row in-process so a bad write never reaches disk silently
        rt = unpack(blob)
        assert rt["sample_id"] == rec["sample_id"] and rt["images"] == rec["images"], \
            f"round-trip mismatch for {rec['sample_id']}"
        records_bytes.append(blob)
        manifest.append({"idx": k, "sample_id": rec["sample_id"],
                         "provenance": rec["provenance"], "bytes": len(blob)})

    out = write_array_record(records_bytes, args.out)
    json.dump(manifest, open(out + ".manifest.json", "w"), indent=2)
    print(f"[etl-local] wrote {len(records_bytes)} records -> {out}")
    print(f"[etl-local] total {sum(m['bytes'] for m in manifest)} msgpack bytes; "
          f"provenance={[m['provenance'] for m in manifest]}")
    print("ETL_LOCAL_DONE")


if __name__ == "__main__":
    main()
