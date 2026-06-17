"""Extract the ~479 rater-scored WOD-E2E val frames into ONE compact tfrecord, so the
internal env can run the FULL processing chain (convert_wod_e2e -> prep -> embeds) on a
known-answer set WITHOUT shipping the ~226 GB raw val tree (item-1 env control).

OWNER-side, autovla env (tensorflow + waymo_open_dataset + compiled end_to_end_driving_data_pb2).
The output preserves each frame's EXACT serialized E2EDFrame bytes, so convert_wod_e2e.py
reads it byte-identically to the live val shards. Rated predicate mirrors
convert_wod_e2e.is_rated / evaluate_waymo_metrics: preference_trajectories[0].preference_score != -1.

Run:
  python jax_ddrive/scripts/extract_rated_val_subset.py \
      --val_tfrecords '/home/kaiwen/data/fast-ddrive/waymo/val/val_*.tfrecord*' \
      --out_tfrecord  /home/kaiwen/data/fast-ddrive/eval/val_rated_479.tfrecord

Verify after:  python jax_ddrive/scripts/probe_dataset.py --raw <out_tfrecord> -n 3
(expect total records == 479, every shown frame rated=True)."""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.environ.get("FASTDDRIVE_REPO", "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive"))


def _is_rated(fr):
    pt = fr.preference_trajectories
    return len(pt) > 0 and pt[0].preference_score != -1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val_tfrecords", required=True, help="glob for the raw val E2EDFrame shards")
    ap.add_argument("--out_tfrecord", required=True, help="output single tfrecord of rated frames")
    ap.add_argument("--max_shards", type=int, default=0, help="cap shards (0=all; for a quick smoke)")
    ap.add_argument("--expect", type=int, default=479, help="expected rated count (warn if mismatch)")
    args = ap.parse_args()

    import tensorflow as tf  # lazy: autovla env
    from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e
    try:
        from tqdm import tqdm
    except Exception:  # tqdm optional
        def tqdm(x, **k):
            return x

    shards = sorted(glob.glob(args.val_tfrecords))
    if args.max_shards:
        shards = shards[: args.max_shards]
    if not shards:
        print(f"[extract] NO shards matched {args.val_tfrecords!r}", file=sys.stderr)
        sys.exit(2)
    print(f"[extract] matched {len(shards)} shard(s); scanning for rated frames ...")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_tfrecord)) or ".", exist_ok=True)
    tmp = args.out_tfrecord + ".tmp"
    seen = kept = 0
    names = set()
    dup = 0
    with tf.io.TFRecordWriter(tmp) as w:
        for sh in tqdm(shards, desc="shards"):
            for raw in tf.data.TFRecordDataset([sh]):
                seen += 1
                b = raw.numpy()
                fr = e2e.E2EDFrame()
                fr.ParseFromString(b)
                if not _is_rated(fr):
                    continue
                nm = fr.frame.context.name
                if nm in names:
                    dup += 1
                    continue
                names.add(nm)
                w.write(b)          # exact original serialized bytes
                kept += 1
    os.replace(tmp, args.out_tfrecord)

    sz = os.path.getsize(args.out_tfrecord) / 1e6
    print("=" * 60)
    print(f"[extract] shards read     : {len(shards)}")
    print(f"[extract] frames seen      : {seen}")
    print(f"[extract] rated frames kept: {kept}")
    print(f"[extract] duplicate names  : {dup}")
    print(f"[extract] out tfrecord     : {args.out_tfrecord} ({sz:.1f} MB)")
    if kept != args.expect:
        print(f"[extract] WARNING: kept {kept} != expected {args.expect} "
              f"(partial glob / --max_shards?)")
    print("EXTRACT_RATED_VAL_SUBSET: PASS" if kept > 0 else "EXTRACT_RATED_VAL_SUBSET: FAIL")


if __name__ == "__main__":
    main()
