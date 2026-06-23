"""Portable record core for the WOD-E2E -> ArrayRecord pipeline (doc 09 §4.1).

This is the *env-portable* heart of **Job ① (Flume ETL)**: the msgpack record schema
(v1) plus the assemble/pack/unpack helpers. It imports NOTHING heavy (no TF, no torch,
no transformers) — only `msgpack` + `numpy` + stdlib — so the exact same code runs:

  * locally, from the already-rendered Fast-dDrive JSON  -> `build_record_from_json`
    (see `etl_local.py`, the local stand-in for the Flume job), and
  * inside google3 Flume, from raw `E2EDFrame` protos -> `build_record_from_proto`
    (see `etl_flume.py`, which calls `assemble_record` with proto-derived fields).

Design principle (doc 09 §4.1 "parity-safe 取舍"): store the **minimal set that keeps
parity**. `prompt_text`/`target_text` are the *raw, already-rendered* strings (exactly
what `prep_train_jax.py` reads from `conversations[*].value`), so the train-time Grain
tokenizer is byte-for-byte identical to the oracle. The raw numeric fields (`ego`,
`intent`, `future_xy`, `rated`) are OPTIONAL — kept only for future prompt-retemplating
or eval metrics, and never consumed by `tokenize_sasd.TokenizeSASD`.

Record schema v1
----------------
  sample_id      : str            (required) frame.context.name; join key + eval GT key
  images         : list[bytes](3) (required) raw JPEG, order (FRONT_LEFT, FRONT, FRONT_RIGHT)
  prompt_text    : str            (required) raw human prompt (pre-tokenize, pre-template)
  target_text    : str            (required) raw gpt JSON target (pre-process_gpt)
  provenance     : str            (required) "pseudo" | "distilled"
  schema_version : int            (required) == SCHEMA_VERSION
  ego            : dict[str->bytes] (optional) raw past-state arrays, float32 .tobytes()
  intent         : int | str      (optional) proto EgoIntent.Intent (int) or nav string
  future_xy      : bytes          (optional) GT future float32[T,2].tobytes()
  rated          : bool           (optional) rater-scored eval subset flag
"""
import msgpack
import numpy as np

SCHEMA_VERSION = 1

# Camera ordering frozen to convert_wod_e2e.FRONT_TRIPLET = [(2,FRONT_LEFT),(1,FRONT),(3,FRONT_RIGHT)].
# The local Fast-dDrive JSON already lists item["image"] in this same FL,F,FR order.
FRONT_TRIPLET_NAMES = ("FRONT_LEFT", "FRONT", "FRONT_RIGHT")

# Required keys, asserted on every record so a malformed sink fails loudly (not silently).
REQUIRED_KEYS = ("sample_id", "images", "prompt_text", "target_text",
                 "provenance", "schema_version")


def assemble_record(sample_id, image_bytes, prompt_text, target_text,
                    provenance="pseudo", *, ego=None, intent=None,
                    future_xy=None, rated=None):
    """Build a schema-v1 record dict from already-extracted fields. Shared by the local
    (JSON) and google3 (proto) front-ends so the on-disk bytes are identical regardless
    of source.

    image_bytes : list of 3 raw JPEG `bytes` in (FL, F, FR) order.
    ego         : optional dict[str -> float32 ndarray]; stored as raw `.tobytes()`.
    future_xy   : optional float32 ndarray [T,2]; stored as raw `.tobytes()`.
    """
    if len(image_bytes) != 3:
        raise ValueError(f"expected 3 images (FL,F,FR), got {len(image_bytes)} for {sample_id}")
    for i, b in enumerate(image_bytes):
        if not isinstance(b, (bytes, bytearray)):
            raise TypeError(f"image[{i}] must be raw bytes, got {type(b).__name__}")
    rec = {
        "sample_id": str(sample_id),
        "images": [bytes(b) for b in image_bytes],
        "prompt_text": str(prompt_text),
        "target_text": str(target_text),
        "provenance": str(provenance),
        "schema_version": SCHEMA_VERSION,
    }
    if ego is not None:
        rec["ego"] = {k: np.asarray(v, np.float32).tobytes() for k, v in ego.items()}
    if intent is not None:
        rec["intent"] = intent
    if future_xy is not None:
        rec["future_xy"] = np.asarray(future_xy, np.float32).tobytes()
    if rated is not None:
        rec["rated"] = bool(rated)
    return rec


def build_record_from_json(item, image_root, *, provenance="pseudo", distilled=None):
    """LOCAL front-end: turn one Fast-dDrive JSON item (= convert_wod_e2e output) into a
    schema-v1 record. Stand-in for the google3 proto front-end; used by `etl_local.py`
    and the parity gate.

    item        : dict with keys "image"(3 paths), "conversations"[human,gpt], "sample_id",
                  and structured fields ("velocity","acceleration","history waypoints",
                  "future waypoints","navigation_command", ...).
    image_root  : dir that item["image"] paths are relative to (the Fast-dDrive repo root).
    distilled   : optional dict {sample_id -> distilled_target_text}; if hit, target_text
                  is taken from it and provenance becomes "distilled" (doc 09 §4.2).
    """
    import os
    sid = item.get("sample_id", "")
    paths = item["image"]
    if len(paths) != 3:
        raise ValueError(f"{sid}: expected 3 image paths, got {len(paths)}")
    image_bytes = []
    for p in paths:
        with open(os.path.join(image_root, p), "rb") as f:
            image_bytes.append(f.read())

    prompt_text = item["conversations"][0]["value"]                 # raw human (pre-template)
    if distilled and sid in distilled:
        target_text, prov = distilled[sid], "distilled"
    else:
        target_text, prov = item["conversations"][1]["value"], provenance   # raw gpt (pre-process_gpt)

    # ---- optional raw fields (best-effort from the local JSON; the real Flume ETL pulls
    #      these from proto past_states/future_states — see etl_flume.py) ----
    ego = None
    if all(k in item for k in ("history waypoints", "velocity", "acceleration")):
        ego = {
            "pos": np.asarray(item["history waypoints"], np.float32),     # [16,2] (x,y)
            "vel": np.asarray(item["velocity"], np.float32),              # [16,2]
            "accel": np.asarray(item["acceleration"], np.float32),        # [16,2]
        }
    future_xy = (np.asarray(item["future waypoints"], np.float32)
                 if "future waypoints" in item else None)                  # [20,2]
    intent = item.get("navigation_command")                                # str locally; int in proto

    return assemble_record(sid, image_bytes, prompt_text, target_text, prov,
                           ego=ego, intent=intent, future_xy=future_xy)


def pack(record):
    """Serialize a record dict to msgpack bytes (one ArrayRecord row). `use_bin_type=True`
    keeps str vs bytes distinct on the wire so JPEG/ego/future blobs round-trip losslessly."""
    _validate(record)
    return msgpack.packb(record, use_bin_type=True)


def unpack(blob):
    """Deserialize one msgpack row back to a record dict (`raw=False` -> str keys/values,
    bytes stay bytes). Inverse of `pack`."""
    rec = msgpack.unpackb(blob, raw=False)
    _validate(rec)
    return rec


def _validate(rec):
    missing = [k for k in REQUIRED_KEYS if k not in rec]
    if missing:
        raise ValueError(f"record missing required keys {missing}")
    if rec["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version {rec['schema_version']} != {SCHEMA_VERSION}")
    if len(rec["images"]) != 3:
        raise ValueError(f"record needs exactly 3 images, got {len(rec['images'])}")
    return rec
