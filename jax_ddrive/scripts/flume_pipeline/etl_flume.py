"""Job ① — the REAL google3 Flume ETL skeleton (doc 09 §4.3). NOT runnable outside google3.

This is the production front-end: it decodes raw `E2EDFrame` protos straight from WOD-E2E
tfrecords and writes schema-v1 msgpack ArrayRecord — with **zero TF and zero torch** (the
Waymo-internal-agent result, doc 09 §2.2). It is intentionally importable as documentation
but raises if you try to run it locally, because:
  * the proto target `//third_party/waymo_open_dataset/protos:end_to_end_driving_data_py_pb2`
    is google3-internal (not on PyPI), and
  * `apache_beam` here means the internal Flume runner + the internal ArrayRecord sink.

The record-building core (`assemble_record`, msgpack `pack`) is shared verbatim with the
local path (`etl_record.py`), so the on-disk bytes are identical regardless of front-end —
that is what lets the local parity gate stand in for this job.

§7 IMPLEMENTATION-TIME CHECKLIST (doc 09) — verify each against the real proto before trusting:
  - image bytes path is the DOUBLE-`frame`: `frame.frame.images[i].image` (JPEG), `.name` = CameraName.
  - `frame.frame.context.name` is the sample_id / join key.
  - past-state arrays live under `frame.past_states.{pos_x,pos_y,accel_x,accel_y,vel_x,vel_y}`.
  - `frame.intent` is the `EgoIntent.Intent` enum; future GT under `frame.future_states`.
  - `render_prompt` / `render_target` must be the TF-free pure-python halves of
    `fast_ddrive/data/convert_wod_e2e.py` (PROMPT_PREFIX, 7-pt @0.5s history, GT-traj indices
    [3,7,11,15,19], meta weak-label rules) — byte-for-byte, or parity vs the oracle breaks.
"""
import io
import sys

import msgpack  # noqa: F401  (documents the wire format; assemble_record/pack do the encoding)

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from etl_record import assemble_record, pack, SCHEMA_VERSION  # noqa: E402,F401

# Frozen camera ordering — same as etl_record.FRONT_TRIPLET_NAMES / convert_wod_e2e.FRONT_TRIPLET.
# Ints are waymo CameraName enum values: FRONT=1, FRONT_LEFT=2, FRONT_RIGHT=3.
FRONT_TRIPLET = [(2, "FRONT_LEFT"), (1, "FRONT"), (3, "FRONT_RIGHT")]

_NOT_GOOGLE3 = (
    "etl_flume.py is the google3 Flume job; it needs the internal E2EDFrame proto + Flume "
    "runner + ArrayRecord sink and cannot run locally. Use etl_local.py for the local path."
)


# ── pure-python WOD-E2E renderers (TF-free halves of convert_wod_e2e.py) ──────────────────
# In google3 these are imported from a shared lib that BOTH convert_wod_e2e.py and this Flume
# job use, so the rendered prompt/target strings match the oracle byte-for-byte. Stubbed here.
def render_prompt(frame):
    raise NotImplementedError(_NOT_GOOGLE3)


def render_target(frame):
    raise NotImplementedError(_NOT_GOOGLE3)


def is_rated(frame):
    raise NotImplementedError(_NOT_GOOGLE3)


def build_record_from_proto(frame, distilled=None):
    """PROTO front-end (google3). Mirror of `etl_record.build_record_from_json` but sourcing
    every field from the `E2EDFrame` proto. Returns msgpack bytes, or None to drop the sample
    (missing front-camera triplet — matches convert_wod_e2e.save_front_images behaviour)."""
    import numpy as np

    sid = frame.frame.context.name
    by_name = {im.name: im.image for im in frame.frame.images}      # name(int) -> JPEG bytes
    if any(cam not in by_name for cam, _ in FRONT_TRIPLET):
        return None
    images = [by_name[cam] for cam, _ in FRONT_TRIPLET]             # (FL, F, FR) JPEG bytes

    prompt_text = render_prompt(frame)
    if distilled and sid in distilled:
        target_text, prov = distilled[sid], "distilled"
    else:
        target_text, prov = render_target(frame), "pseudo"

    ps = frame.past_states
    ego = {k: np.asarray(getattr(ps, k), np.float32)
           for k in ("pos_x", "pos_y", "accel_x", "accel_y", "vel_x", "vel_y")}
    future_xy = np.asarray([[s.pos_x, s.pos_y] for s in frame.future_states], np.float32)

    rec = assemble_record(sid, images, prompt_text, target_text, prov,
                          ego=ego, intent=int(frame.intent),
                          future_xy=future_xy, rated=is_rated(frame))
    return pack(rec)


def build_pipeline(input_glob, output_path, distilled=None):
    """The Flume/Beam pipeline (doc 09 §4.3). Internal runner + ArrayRecord sink names are
    placeholders — wire to the google3 equivalents at implementation time."""
    import apache_beam as beam
    from third_party.waymo_open_dataset.protos import (   # type: ignore  # google3-internal
        end_to_end_driving_data_pb2 as e2e)

    def _to_record(frame):
        return build_record_from_proto(frame, distilled)

    return (
        "WodE2eToArrayRecord",
        lambda p: (
            p
            | "Read" >> beam.io.ReadFromTFRecord(input_glob)
            | "ParseProto" >> beam.Map(e2e.E2EDFrame.FromString)
            | "ToRecord" >> beam.Map(_to_record)
            | "DropEmpty" >> beam.Filter(lambda x: x is not None)
            | "WriteAR" >> _WriteToArrayRecord(output_path)         # internal sink API
        ),
    )


class _WriteToArrayRecord(object):                                  # placeholder; bind to google3 AR sink
    """Stand-in for the internal ArrayRecord Beam sink (e.g. a `beam.PTransform` wrapping the
    google3 ArrayRecord IO). Bind to the real sink at implementation time."""
    def __init__(self, path):
        self.path = path


if __name__ == "__main__":
    raise SystemExit(_NOT_GOOGLE3)
