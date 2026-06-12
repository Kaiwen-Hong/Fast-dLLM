#!/usr/bin/env python3
"""Teacher-distill merge (Stage B): swap pseudo text sections for teacher-generated ones.

Raw WOD-E2E has no text labels, so ``convert_wod_e2e.py --with_target`` produces
*weak* labels: real GT ``trajectory`` but pseudo ``critical_objects`` (all "no"),
templated ``explanation`` and intent-derived ``future_meta_behavior``.

This script upgrades those three text sections using the released Fast-dDrive
checkpoint's predictions (run via ``eval/batch_inference.py``), while keeping the
**real GT trajectory** untouched. Output is the same Fast-dDrive training JSON
schema, ready to re-tokenize into NPZ/Parquet (Stage C).

Join key: ``sample_id``. We start from the ORIGINAL training sample (so every
field, including the real-GT ``trajectory`` already inside its gpt JSON, is
preserved byte-for-byte) and replace only critical_objects / explanation /
future_meta_behavior with the teacher's parsed output.

Usage::

    python fast_ddrive/data/merge_distilled_labels.py \
        --teacher_pred /home/kaiwen/data/fast-ddrive/train/distill_teacher_10/predictions.json \
        --orig_json    /home/kaiwen/data/fast-ddrive/train/train_targets.json \
        --out_json     /home/kaiwen/data/fast-ddrive/train/train_targets_distilled_10.json
"""
import argparse
import json
import sys

# critical_objects + explanation are ALWAYS taken from the teacher (it grounds them
# in the image; the converter pseudo is all-"no" / templated). ``trajectory`` is
# never touched (real GT stays). ``future_meta_behavior`` is handled per --fmb_mode:
# its ``longitudinal`` sub-field is better kept from the GT-derived label (computed
# from the real future trajectory: speed up / slow down / come to stop), while its
# ``lateral`` sub-field is better taken from the teacher (canonical vocab the
# converter cannot produce: lane follow / lane change / yield).
ALWAYS_TEACHER = ("critical_objects", "explanation")


def load_predictions(path):
    """Return {sample_id: parsed_teacher_obj_or_None} from a batch_inference predictions.json."""
    with open(path) as f:
        blob = json.load(f)
    # predictions.json is {"metadata":..., "predictions":[entry,...]} or a bare list.
    preds = blob["predictions"] if isinstance(blob, dict) and "predictions" in blob else blob
    out = {}
    for e in preds:
        out[e["sample_id"]] = {
            "parsed": e.get("model_output_parsed"),
            "raw": e.get("model_output_raw"),
            "error": e.get("error"),
        }
    return out


def section_has_yes(critical_objects):
    """True if the teacher flagged at least one critical object (vs the all-'no' pseudo)."""
    if not isinstance(critical_objects, dict):
        return False
    return any(str(v).strip().lower().startswith("y") for v in critical_objects.values())


def main():
    ap = argparse.ArgumentParser(description="Merge teacher text sections into training JSON (keep real GT trajectory).")
    ap.add_argument("--teacher_pred", required=True, help="predictions.json from eval/batch_inference.py")
    ap.add_argument("--orig_json", required=True, help="Original Fast-dDrive training JSON (with real-GT trajectory + pseudo text).")
    ap.add_argument("--out_json", required=True, help="Output distilled training JSON.")
    ap.add_argument("--fmb_mode", choices=("hybrid", "teacher", "gt"), default="hybrid",
                    help="future_meta_behavior policy. hybrid (default): GT-derived longitudinal + "
                         "teacher lateral. teacher: take the teacher's whole fmb. gt: keep the pseudo fmb.")
    ap.add_argument("--strict", action="store_true",
                    help="Error out if a teacher section is missing/unparseable instead of falling back to the pseudo label.")
    args = ap.parse_args()

    teacher = load_predictions(args.teacher_pred)
    with open(args.orig_json) as f:
        orig = json.load(f)
    orig_by_id = {s["sample_id"]: s for s in orig}

    # Only process the samples the teacher actually ran on (e.g. the 10-sample subset).
    ids = [sid for sid in teacher if sid in orig_by_id]

    stats = {
        "teacher_samples": len(teacher),
        "matched": len(ids),
        "fmb_mode": args.fmb_mode,
        "teacher_parse_fail": 0,
        "fully_distilled": 0,
        "partial_fallback": 0,
        "co_with_any_yes": 0,
        "per_section_replaced": {"critical_objects": 0, "explanation": 0, "fmb_lateral": 0},
        "fmb_longitudinal_from_gt": 0,
    }
    missing_ids = [sid for sid in teacher if sid not in orig_by_id]

    out_samples = []
    for sid in ids:
        sample = json.loads(json.dumps(orig_by_id[sid]))  # deep copy
        gpt = sample["conversations"][1]
        orig_obj = json.loads(gpt["value"]) if gpt["value"] else {}

        tinfo = teacher[sid]
        tobj = tinfo["parsed"]
        if not isinstance(tobj, dict):
            stats["teacher_parse_fail"] += 1
            if args.strict:
                sys.exit(f"[strict] teacher output unparseable for sample_id={sid} (error={tinfo.get('error')})")
            # keep original pseudo entirely
            sample["_distill"] = {"status": "teacher_parse_fail", "error": (tinfo.get("error") or "")[:200]}
            out_samples.append(sample)
            continue

        distilled = dict(orig_obj)
        replaced = []
        # critical_objects + explanation: always from teacher.
        for sec in ALWAYS_TEACHER:
            if sec in tobj and tobj[sec] not in (None, "", {}):
                distilled[sec] = tobj[sec]
                replaced.append(sec)
                stats["per_section_replaced"][sec] += 1
            elif args.strict:
                sys.exit(f"[strict] teacher missing section '{sec}' for sample_id={sid}")

        # future_meta_behavior: per policy.
        orig_fmb = orig_obj.get("future_meta_behavior", {}) or {}
        teacher_fmb = tobj.get("future_meta_behavior", {}) or {}
        if args.fmb_mode == "teacher":
            if teacher_fmb:
                distilled["future_meta_behavior"] = teacher_fmb
                replaced.append("fmb")
            elif args.strict:
                sys.exit(f"[strict] teacher missing future_meta_behavior for sample_id={sid}")
        elif args.fmb_mode == "hybrid":
            fmb = dict(orig_fmb)  # GT longitudinal (+ GT lateral as fallback)
            tl = teacher_fmb.get("lateral")
            if tl not in (None, ""):
                fmb["lateral"] = tl
                replaced.append("fmb_lateral")
                stats["per_section_replaced"]["fmb_lateral"] += 1
            elif args.strict:
                sys.exit(f"[strict] teacher missing future_meta_behavior.lateral for sample_id={sid}")
            if "longitudinal" in orig_fmb:
                stats["fmb_longitudinal_from_gt"] += 1
            distilled["future_meta_behavior"] = fmb
        # else fmb_mode == "gt": keep orig_obj's future_meta_behavior unchanged.
        # trajectory stays = orig_obj["trajectory"] (real GT), untouched.

        gpt["value"] = json.dumps(distilled, ensure_ascii=False)
        if section_has_yes(distilled.get("critical_objects")):
            stats["co_with_any_yes"] += 1
        # "fully distilled" = both always-teacher sections + the fmb policy applied.
        expected = list(ALWAYS_TEACHER) + (["fmb_lateral"] if args.fmb_mode == "hybrid"
                                           else (["fmb"] if args.fmb_mode == "teacher" else []))
        if all(r in replaced for r in expected):
            stats["fully_distilled"] += 1
        else:
            stats["partial_fallback"] += 1
        sample["_distill"] = {"status": "ok", "replaced": replaced}
        out_samples.append(sample)

    with open(args.out_json, "w") as f:
        json.dump(out_samples, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"[distill-merge] wrote {len(out_samples)} samples -> {args.out_json}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if missing_ids:
        print(f"[warn] {len(missing_ids)} teacher sample_ids not found in orig_json (first 3): {missing_ids[:3]}")
    if stats["co_with_any_yes"] == 0 and stats["matched"] > 0:
        print("[warn] no sample got a critical_objects 'yes' — teacher may be degenerate or all scenes truly empty.")
    print("MERGE_DONE")


if __name__ == "__main__":
    main()
