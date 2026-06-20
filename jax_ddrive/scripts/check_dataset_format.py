"""Standalone FORMAT-CHECKER for a PROCESSED Fast-dDrive SASD dataset directory.

Asserts that a processed dataset dir (ArrayRecord v2 OR Parquet v1) is self-consistent
against the canonical schema -- **without** needing a parallel parquet source to diff
against (cf. ``verify_ar_v2_dataset.py``, which byte-compares AR vs a source parquet set we
do not have internally).  This script keeps only the SCHEMA self-checks and drops the
"compare vs source parquet row" part.

Checks (all CPU, no GPU, no model):
  (1) COUNT     : decodable records == dataset_info_<split>.json ``num_samples`` sidecar.
  (2) FIELDS    : the 12 base arrays + sample_id/L/n_blocks present; for v2 also
                  ``image_embeds`` (OPTIONAL: absent=pixels-only, present=embeds) -- whose
                  presence is UNIFORM across ALL shards
                  (mixed presence => FAIL, mirroring grain_pipeline's has_embeds guard).
  (3) DTYPES    : exact dtype per ARRAY_DTYPES (+ image_embeds bfloat16) on sampled records.
  (4) SHAPES    : per-record RESOLUTION-DERIVED shapes -- [L]/[3,L]/(N,1176)/(3,3) with
                  N=sum(t*h*w) from image_grid_thw and L uniform across records (no hardcoded
                  L); image_embeds (when present) (N//4,2048); n_blocks scalar ==
                  block_alpha/beta length, L%32==0.
  (5) CROSS-FIELD: image_pad runs==3 with len==(t*h*w)/4 and sum==pixel/4==N//4 (derived,
                  not literal 168); vision_mask
                  == isin(input_ids,{image_pad,vision_start,vision_pad}); labels!=-100 only on
                  the assistant span bracketed by 151644/77091/198 .. 151645 with labels==ids
                  there; weight_vec values subset of {1.0,1.5,2.0,3.0}.
  (6) FINITE    : image_embeds + pixel_values all finite; sample_id non-empty.

Reuses ``ar_dataset.decode_example`` / ``parquet_dataset.decode_row`` for decoding and
``prep_to_parquet.ARRAY_DTYPES`` for the canonical dtype table.

  python check_dataset_format.py --dir <AR_or_parquet_dir> \
        [--split auto] [--expect v2|parquet|auto] [--samples 32] [--strict]

Prints one PASS/FAIL line per check, then a final CHECK_PASS / CHECK_FAIL token plus the
failure count.  Exit code 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

# Repo root on sys.path so ``ddrive_jax.*`` imports resolve (matches verify_ar_v2_dataset.py
# / prep_train_jax.py convention; parameterized for portability).
REPO = os.environ.get("FASTDDRIVE_REPO", "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
SNAP_DEFAULT = os.environ.get(
    "FASTDDRIVE_SNAP",
    "/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
    "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f",
)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ddrive_jax.convert.prep_to_parquet import ARRAY_DTYPES, ARRAY_FIELDS  # noqa: E402

# --- canonical schema constants (SSOT mirrored from the task spec / verify_ar_round2.py) ---
# NOTE: shapes/token-counts are RESOLUTION-DERIVED per record from image_grid_thw (so both the
# 168/L=1184 embeds AR and the 720/L=1856 pixels-only AR validate with one code path); the
# only genuinely constant dataset-family facts kept here are the 1176-dim Qwen2-VL patch
# feature and the 3-image context grid.
PIXEL_FEAT = 1176            # pixel_values cols (Qwen2-VL patch feature dim -- constant)
GRID_SHAPE = (3, 3)          # image_grid_thw -- 3-image context, constant for this family
EMB_FEAT = 2048              # image_embeds feature dim (rows derived = sum(t*h*w)//4)

MASK_ID = 151665
IM_END = 151645
IMAGE_PAD = 151655
VISION_START = 151652
VISION_PAD = 151654
IM_START = 151644            # assistant-span bracket: <|im_start|>
ASSIST_TAG = 77091          # "assistant" token id
NEWLINE = 198               # "\n" id
SECTION_WEIGHTS = {1.0, 1.5, 2.0, 3.0}

# dtype each array must decode to (numpy dtype) -- base 12 from ARRAY_DTYPES + image_embeds.
EXPECT_DTYPES = {k: np.dtype(v) for k, v in ARRAY_DTYPES.items()}

# Fields whose canonical shape is RESOLUTION-DERIVED per record (built from L_ref / N below).
L_FIELDS = ("input_ids", "labels", "rbi", "turn", "scaffold", "vision_mask", "weight_vec")


def _expected_shapes(L, N):
    """Per-record canonical shapes derived from sequence length L and pixel-row count N.

    L  = input_ids.shape[0] (uniform across records, asserted separately).
    N  = sum(t*h*w) over image_grid_thw rows == pixel_values row count.
    """
    shp = {f: (L,) for f in L_FIELDS}
    shp["position_ids"] = (3, L)
    shp["pixel_values"] = (N, PIXEL_FEAT)
    shp["image_grid_thw"] = GRID_SHAPE
    return shp


# ============================================================================ helpers =====
def _bfloat16_dtype():
    """ml_dtypes.bfloat16 numpy dtype (image_embeds storage dtype). Lazy import."""
    import ml_dtypes
    return np.dtype(ml_dtypes.bfloat16)


def _runs(ids):
    """[(token_id, start, length)] of maximal constant runs (from verify_ar_round2._runs)."""
    out, i, n = [], 0, len(ids)
    while i < n:
        j = i
        while j < n and ids[j] == ids[i]:
            j += 1
        out.append((int(ids[i]), i, j - i))
        i = j
    return out


def _detect_split(data_dir):
    """Return (kind, split, shard_paths) where kind in {'arrayrecord','parquet'}.

    Auto-detects format by extension and split by the ``<split>-*`` shard prefix actually on
    disk.  Errors if both formats coexist (ambiguous, mirroring grain_pipeline) or none found.
    """
    ar = sorted(glob.glob(os.path.join(data_dir, "*.arrayrecord")))
    pqs = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if ar and pqs:
        raise ValueError(f"both arrayrecord and parquet shards under {data_dir!r} -- ambiguous")
    if ar:
        kind, paths = "arrayrecord", ar
    elif pqs:
        kind, paths = "parquet", pqs
    else:
        raise FileNotFoundError(f"no *.arrayrecord or *.parquet shards under {data_dir!r}")
    # split = prefix before the first '-NNNNN-of-' shard token.
    splits = sorted({os.path.basename(p).split("-")[0] for p in paths})
    if len(splits) != 1:
        raise ValueError(f"multiple splits {splits} under {data_dir!r}; one split per dir expected")
    split = splits[0]
    return kind, split, paths


# ============================================================ per-format decode adapters ===
class _ARAdapter:
    """Lazy ArrayRecord random-access decode via ar_dataset.decode_example + per-shard probe."""

    def __init__(self, paths):
        from array_record.python.array_record_data_source import ArrayRecordDataSource
        from ddrive_jax.data import ar_dataset
        self._decode = ar_dataset.decode_example
        self._paths = list(paths)
        self._ds = ArrayRecordDataSource(self._paths)
        # global index bounds per shard (for per-shard presence probing) -----------------
        self._bounds = [0]
        for p in self._paths:
            self._bounds.append(self._bounds[-1] + len(ArrayRecordDataSource([p])))

    def __len__(self):
        return len(self._ds)

    def get(self, i):
        return self._decode(self._ds[i])

    def shard_first_indices(self):
        """One representative global index per shard (first record of each shard)."""
        return [self._bounds[s] for s in range(len(self._paths)) if self._bounds[s] < len(self)]


class _PQAdapter:
    """Eager-ish Parquet random access via parquet_dataset.decode_row + a flat index."""

    def __init__(self, paths):
        import pyarrow.parquet as pq
        from ddrive_jax.data import parquet_dataset
        self._pq = pq
        self._decode = parquet_dataset.decode_row
        self._paths = list(paths)
        self._counts = [pq.read_metadata(p).num_rows for p in self._paths]
        self._bounds = list(np.cumsum([0] + self._counts))
        self._cache = {}  # shard_idx -> pyarrow.Table

    def __len__(self):
        return self._bounds[-1]

    def _locate(self, i):
        s = int(np.searchsorted(self._bounds, i, side="right") - 1)
        return s, i - self._bounds[s]

    def _table(self, s):
        if s not in self._cache:
            self._cache[s] = self._pq.read_table(self._paths[s])
        return self._cache[s]

    def get(self, i):
        s, local = self._locate(i)
        return self._decode(self._table(s).slice(local, 1).to_pylist()[0])

    def shard_first_indices(self):
        return [self._bounds[s] for s in range(len(self._paths)) if self._counts[s] > 0]


# ================================================================================ checks ===
def _read_num_samples(data_dir, split):
    """num_samples from dataset_info_<split>.json sidecar (None if missing)."""
    p = os.path.join(data_dir, f"dataset_info_{split}.json")
    if not os.path.exists(p):
        return None, p
    return json.load(open(p)).get("num_samples"), p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="processed dataset dir (AR v2 or parquet v1)")
    ap.add_argument("--split", default="auto",
                    help="split name; 'auto' = infer from shard filenames")
    ap.add_argument("--expect", choices=["v2", "parquet", "pixels", "auto"], default="auto",
                    help="expected format: v2 (AR+image_embeds), parquet (v1, no embeds), "
                         "pixels (AR or parquet WITHOUT image_embeds), or auto-detect")
    ap.add_argument("--samples", type=int, default=32,
                    help="# deterministic records to deep-check (dtype/shape/cross-field)")
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings (e.g. missing sidecar) as failures too")
    args = ap.parse_args()

    # Lazy-import / pin TF to CPU so importing/--help never grabs the GPU (ar_dataset does
    # this internally on first decode; we pin here too as a belt-and-braces guard).
    # (TF only imported inside _ARAdapter via ar_dataset; no top-level tf/jax import.)

    failures = 0
    warnings = 0

    def report(name, ok, detail=""):
        nonlocal failures
        flag = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{flag}] {name}" + (f"  ::  {detail}" if detail else ""))

    def warn(name, detail=""):
        nonlocal warnings, failures
        print(f"[WARN] {name}" + (f"  ::  {detail}" if detail else ""))
        warnings += 1
        if args.strict:
            failures += 1

    data_dir = args.dir
    print(f"# check_dataset_format  dir={data_dir}  expect={args.expect}  "
          f"samples={args.samples}  strict={args.strict}")
    print(f"# repo={REPO}")

    # --- format / split detection -------------------------------------------------------
    try:
        kind, det_split, paths = _detect_split(data_dir)
    except Exception as e:  # noqa: BLE001
        report("detect: format+split", False, repr(e))
        print(f"CHECK_FAIL failures={failures}")
        sys.exit(1)
    split = det_split if args.split == "auto" else args.split
    is_v2_format = (kind == "arrayrecord")
    print(f"# detected kind={kind} split={split} shards={len(paths)}")

    # reconcile --expect with detected kind (container axis only; embeds axis handled below).
    if args.expect == "v2" and not is_v2_format:
        report("expect: v2 requires arrayrecord shards", False, f"got kind={kind}")
    elif args.expect == "parquet" and is_v2_format:
        report("expect: parquet requires .parquet shards", False, f"got kind={kind}")
    else:
        # 'pixels' is container-agnostic (AR or parquet); its embeds-absent intent is asserted
        # in the FIELD-PRESENCE reconciliation once has_embeds is known.
        report("expect: format matches --expect", True, f"kind={kind}")

    # --- build adapter ------------------------------------------------------------------
    try:
        adapter = _ARAdapter(paths) if is_v2_format else _PQAdapter(paths)
    except Exception as e:  # noqa: BLE001
        report("decode: open dataset", False, repr(e))
        print(f"CHECK_FAIL failures={failures}")
        sys.exit(1)
    n_records = len(adapter)

    # ===================================================================== (1) COUNT ======
    num_samples, sidecar = _read_num_samples(data_dir, split)
    if num_samples is None:
        warn("count: dataset_info sidecar present", f"missing {os.path.basename(sidecar)}")
        report("count: decodable records (no sidecar to match)", n_records > 0,
               f"records={n_records}")
    else:
        report("count: records == dataset_info num_samples", n_records == int(num_samples),
               f"records={n_records} sidecar_num_samples={num_samples}")

    # ============================================ (2) FIELD PRESENCE + uniform embeds ======
    # Probe the FIRST record of every shard to assert uniform image_embeds presence (mirrors
    # grain_pipeline: mixed presence across shards => FAIL).
    base_keys = set(ARRAY_FIELDS)
    scalar_keys = {"sample_id", "L", "n_blocks"}
    shard_probe_idx = adapter.shard_first_indices()
    embeds_presence = set()
    missing_base = None
    missing_scalar = None
    probe_ok = True
    for gi in shard_probe_idx:
        try:
            rec = adapter.get(gi)
        except Exception as e:  # noqa: BLE001
            probe_ok = False
            report("fields: shard-probe decode", False, f"idx={gi}: {e!r}")
            break
        mb = base_keys - set(rec)
        ms = scalar_keys - set(rec)
        if mb and missing_base is None:
            missing_base = (gi, sorted(mb))
        if ms and missing_scalar is None:
            missing_scalar = (gi, sorted(ms))
        embeds_presence.add("image_embeds" in rec)

    if probe_ok:
        report("fields: 12 base arrays present (all shards)", missing_base is None,
               "" if missing_base is None else f"idx={missing_base[0]} missing={missing_base[1]}")
        report("fields: sample_id/L/n_blocks present (all shards)", missing_scalar is None,
               "" if missing_scalar is None
               else f"idx={missing_scalar[0]} missing={missing_scalar[1]}")
        # uniform presence: exactly one of {True}/{False} across shards.
        uniform = (len(embeds_presence) == 1)
        report("fields: image_embeds presence uniform across shards", uniform,
               f"presence_set={sorted(embeds_presence)}")
        has_embeds = uniform and (True in embeds_presence)
        # reconcile with --expect / format
        if args.expect == "v2":
            report("fields: v2 expects image_embeds present", has_embeds,
                   f"has_embeds={has_embeds}")
        elif args.expect == "parquet":
            report("fields: parquet (v1) expects NO image_embeds", not has_embeds,
                   f"has_embeds={has_embeds}")
        elif args.expect == "pixels":
            report("fields: pixels expects NO image_embeds (any container)", not has_embeds,
                   f"has_embeds={has_embeds} kind={kind}")
        else:
            print(f"[INFO] auto: has_embeds={has_embeds} (kind={kind})")
    else:
        has_embeds = False

    # =============================== deterministic deep-check sample set (3,4,5,6) =========
    k = max(1, min(args.samples, n_records))
    deep_idx = sorted(set(
        [0, n_records - 1]
        + np.linspace(0, n_records - 1, num=k, dtype=int).tolist()
    ))

    # accumulators -- one verdict per check, AND-reduced over the sampled records.
    dtype_bad = []       # (idx, field, got, want)
    shape_bad = []       # (idx, field, got, want)
    nblocks_bad = []     # (idx, detail)
    Lmod_bad = []        # (idx, L)
    Luniform_bad = []    # (idx, L, L_ref) -- L differs from first deep-checked record
    Lscalar_bad = []     # (idx, scalar_L, L) -- sidecar scalar L != input_ids length
    L_ref = None         # uniform per-record sequence length, captured from first record
    runs_bad = []        # (idx, detail)
    vmask_bad = []       # (idx,)
    assist_bad = []      # (idx, detail)
    wvocab_bad = []      # (idx, set)
    finite_bad = []      # (idx, field)
    sid_bad = []         # (idx,)

    bf16 = _bfloat16_dtype()

    for gi in deep_idx:
        rec = adapter.get(gi)

        # ---- (3) DTYPES --------------------------------------------------------------
        for f, want in EXPECT_DTYPES.items():
            if f in rec and rec[f].dtype != want:
                dtype_bad.append((gi, f, str(rec[f].dtype), str(want)))
        # NB: compare bf16 by dtype.name — TF's .numpy() bfloat16 and the standalone
        # ml_dtypes.bfloat16 are two DISTINCT numpy extension dtypes (both named 'bfloat16')
        # that compare unequal by object identity; .name is the robust, decoder-agnostic check.
        if has_embeds and "image_embeds" in rec and rec["image_embeds"].dtype.name != bf16.name:
            dtype_bad.append((gi, "image_embeds", str(rec["image_embeds"].dtype), str(bf16)))

        # ---- (4) SHAPES (resolution-derived) -----------------------------------------
        ids = rec["input_ids"]
        L = int(ids.shape[0])
        grid = rec["image_grid_thw"]
        N = sum(int(t * h * w) for t, h, w in grid)   # pixel_values row count
        n_tok = N // 4                                # image-token / image_pad count
        if L_ref is None:
            L_ref = L                                 # capture uniform L from the first record
        # uniform-L guard: every record's L (and its sidecar scalar L) must equal L_ref.
        if L != L_ref:
            Luniform_bad.append((gi, L, L_ref))
        scalar_L = int(rec["L"]) if "L" in rec else L
        if scalar_L != L:
            Lscalar_bad.append((gi, scalar_L, L))
        exp_shapes = _expected_shapes(L, N)
        for f, want in exp_shapes.items():
            if f in rec and tuple(rec[f].shape) != want:
                shape_bad.append((gi, f, tuple(rec[f].shape), want))
        emb_shape = (n_tok, EMB_FEAT)
        if has_embeds and "image_embeds" in rec and tuple(rec["image_embeds"].shape) != emb_shape:
            shape_bad.append((gi, "image_embeds", tuple(rec["image_embeds"].shape), emb_shape))
        if L % 32 != 0:
            Lmod_bad.append((gi, L))
        # n_blocks scalar == len(block_alpha) == len(block_beta)
        nb = int(rec["n_blocks"])
        ba, bb = rec["block_alpha"], rec["block_beta"]
        if not (ba.shape == bb.shape == (nb,)):
            nblocks_bad.append((gi, f"n_blocks={nb} alpha={ba.shape} beta={bb.shape}"))

        # ---- (5) CROSS-FIELD (resolution-derived) ------------------------------------
        # image_pad runs: one run per grid image, length (t*h*w)//4, summing to n_tok == N//4.
        pad_runs = [(s, ln) for t, s, ln in _runs(list(ids)) if t == IMAGE_PAD]
        per_img = [int(t * h * w) // 4 for (t, h, w) in grid]
        sum_runs = sum(ln for _, ln in pad_runs)
        n_pix = int(rec["pixel_values"].shape[0])
        run_ok = (
            len(pad_runs) == len(grid) == 3
            and [ln for _, ln in pad_runs] == per_img
            and sum_runs == n_pix // 4 == n_tok
        )
        if not run_ok:
            runs_bad.append((gi, f"n_runs={len(pad_runs)} run_lens={[ln for _, ln in pad_runs]} "
                                 f"exp={per_img} sum={sum_runs} pix//4={n_pix // 4} n_tok={n_tok}"))
        # vision_mask == isin(ids, {image_pad, vision_start, vision_pad})
        vm_ref = np.isin(ids, [IMAGE_PAD, VISION_START, VISION_PAD])
        if not np.array_equal(rec["vision_mask"], vm_ref):
            vmask_bad.append((gi,))
        # labels: nonzero (!= -100) only on assistant span bracketed by IM_START/77091/198..IM_END
        labels = rec["labels"]
        resp = np.where(labels != -100)[0]
        if resp.size == 0:
            assist_bad.append((gi, "no labels != -100"))
        else:
            rs, re_ = int(resp[0]), int(resp[-1]) + 1
            contiguous = (re_ - rs) == resp.size  # the response span is a single contiguous run
            head_ok = (rs >= 3 and int(ids[rs - 3]) == IM_START
                       and int(ids[rs - 2]) == ASSIST_TAG and int(ids[rs - 1]) == NEWLINE)
            tail_ok = int(ids[re_ - 1]) == IM_END
            eq_ok = np.array_equal(labels[rs:re_], ids[rs:re_])
            if not (contiguous and head_ok and tail_ok and eq_ok):
                assist_bad.append((gi, f"contig={contiguous} head={head_ok} "
                                       f"tail={tail_ok} labels==ids={eq_ok}"))
        # weight_vec values subset of section weights
        wset = set(np.unique(rec["weight_vec"]).astype(np.float32).tolist())
        if not wset <= SECTION_WEIGHTS:
            wvocab_bad.append((gi, sorted(wset)))

        # ---- (6) FINITENESS + non-empty sample_id ------------------------------------
        if not np.isfinite(rec["pixel_values"].astype(np.float32)).all():
            finite_bad.append((gi, "pixel_values"))
        if has_embeds and "image_embeds" in rec:
            if not np.isfinite(rec["image_embeds"].astype(np.float32)).all():
                finite_bad.append((gi, "image_embeds"))
        sid = rec.get("sample_id", "")
        if not (isinstance(sid, str) and len(sid) > 0):
            sid_bad.append((gi,))

    n_deep = len(deep_idx)
    report(f"dtype: exact ARRAY_DTYPES (+image_embeds bf16) over {n_deep} records",
           not dtype_bad, "" if not dtype_bad else f"{len(dtype_bad)} bad e.g. {dtype_bad[0]}")
    report(f"shape: canonical shapes over {n_deep} records",
           not shape_bad, "" if not shape_bad else f"{len(shape_bad)} bad e.g. {shape_bad[0]}")
    report(f"shape: L%32==0 over {n_deep} records",
           not Lmod_bad, "" if not Lmod_bad else f"{len(Lmod_bad)} bad e.g. {Lmod_bad[0]}")
    report(f"shape: L uniform (==L_ref={L_ref}) across {n_deep} records",
           not Luniform_bad, "" if not Luniform_bad else
           f"{len(Luniform_bad)} bad e.g. {Luniform_bad[0]} (mixed L => FAIL)")
    report(f"shape: sidecar scalar L == input_ids length over {n_deep} records",
           not Lscalar_bad, "" if not Lscalar_bad else f"{len(Lscalar_bad)} bad e.g. {Lscalar_bad[0]}")
    report(f"shape: n_blocks==len(block_alpha/beta) over {n_deep} records",
           not nblocks_bad, "" if not nblocks_bad else f"{len(nblocks_bad)} bad e.g. {nblocks_bad[0]}")
    report(f"cross: image_pad 3 runs, len==(t*h*w)/4, sum==pix/4==N//4 (derived) over {n_deep}",
           not runs_bad, "" if not runs_bad else f"{len(runs_bad)} bad e.g. {runs_bad[0]}")
    report(f"cross: vision_mask==isin(ids,{{pad,vstart,vpad}}) over {n_deep}",
           not vmask_bad, "" if not vmask_bad else f"{len(vmask_bad)} bad e.g. {vmask_bad[0]}")
    report(f"cross: labels on assistant span [151644/77091/198..151645], labels==ids over {n_deep}",
           not assist_bad, "" if not assist_bad else f"{len(assist_bad)} bad e.g. {assist_bad[0]}")
    report(f"cross: weight_vec subset of {{1.0,1.5,2.0,3.0}} over {n_deep}",
           not wvocab_bad, "" if not wvocab_bad else f"{len(wvocab_bad)} bad e.g. {wvocab_bad[0]}")
    report(f"finite: pixel_values{' + image_embeds (bf16)' if has_embeds else ' (pixels-only)'} "
           f"all finite over {n_deep}",
           not finite_bad, "" if not finite_bad else f"{len(finite_bad)} bad e.g. {finite_bad[0]}")
    report(f"finite: sample_id non-empty over {n_deep}",
           not sid_bad, "" if not sid_bad else f"{len(sid_bad)} bad e.g. {sid_bad[0]}")

    # ===================================================================== verdict ========
    print(f"# warnings={warnings} (strict={args.strict})")
    if failures == 0:
        print(f"CHECK_PASS failures=0 (deep_checked={n_deep}, records={n_records})")
        sys.exit(0)
    else:
        print(f"CHECK_FAIL failures={failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
