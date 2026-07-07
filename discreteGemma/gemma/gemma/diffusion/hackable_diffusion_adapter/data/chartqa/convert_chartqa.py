# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Convert HF ChartQA to the internal record format (design doc §5 B4).

Produces, under ``schema.LOCAL_DATA_ROOT``:
  * ``human_train-*.arrayrecord`` — train subset (ArrayRecord of tf.Example,
    mirroring the internal ``human_train@90`` layout);
  * ``human_val-*.arrayrecord``   — 256-example eval subset (``slice_stop=256``);
  * ``chartqa_toy.bagz``          — 10-example toy set (Bagz; exercises the
    trajectory-side reader and the C6 padding tests).

Each record = serialized ``tf.train.Example`` with features
``image/encoded`` (PNG bytes) / ``question`` / ``answer`` [internal·reply].

Run:  python convert_chartqa.py [--n_train 2000] [--n_val 256] [--n_toy 10]
"""

import argparse
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("HF_HOME", "/home/kaiwen/data/huggingface")

from array_record.python import array_record_module as arm  # noqa: E402
import bagz  # noqa: E402
import tensorflow as tf  # noqa: E402

from gemma.diffusion.hackable_diffusion_adapter.data.chartqa import schema  # noqa: E402

_SHARD_SIZE = 1024  # records per ArrayRecord shard


def make_example(image_png: bytes, question: str, answer: str) -> bytes:
  feats = {
      schema.FEATURE_IMAGE: image_png,
      schema.FEATURE_QUESTION: question.encode("utf-8"),
      schema.FEATURE_ANSWER: answer.encode("utf-8"),
  }
  ex = tf.train.Example(
      features=tf.train.Features(
          feature={
              k: tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
              for k, v in feats.items()
          }
      )
  )
  return ex.SerializeToString()


def _to_encoded_bytes(img) -> bytes:
  """Coerce an HF image cell (PIL / bytes / dict / path) to PNG/JPEG bytes."""
  from PIL import Image  # pylint: disable=g-import-not-at-top

  if isinstance(img, dict):
    img = img.get("bytes") or img.get("path")
  if isinstance(img, (bytes, bytearray)):
    return bytes(img)  # already encoded — schema accepts JPEG or PNG bytes
  if isinstance(img, str):
    with open(img, "rb") as f:
      return f.read()
  assert isinstance(img, Image.Image), type(img)
  buf = io.BytesIO()
  img.convert("RGB").save(buf, format="PNG")
  return buf.getvalue()


def iter_hf(split: str, want: int, human_only: bool = True):
  import datasets  # pylint: disable=g-import-not-at-top

  ds = datasets.load_dataset("ahmed-masry/ChartQA", split=split)
  n = 0
  for ex in ds:
    if human_only and str(ex.get("type", "human")).lower() not in ("human", ""):
      continue
    q = str(ex.get("query", ex.get("question", "")))
    a = str(ex["label"][0] if isinstance(ex["label"], list) else ex["label"])
    yield make_example(_to_encoded_bytes(ex["image"]), q, a)
    n += 1
    if n >= want:
      return


def write_arrayrecord_shards(records: list[bytes], out_dir: str, pattern: str):
  n_shards = max(1, (len(records) + _SHARD_SIZE - 1) // _SHARD_SIZE)
  paths = []
  for i in range(n_shards):
    path = os.path.join(out_dir, pattern.format(i=i, n=n_shards))
    w = arm.ArrayRecordWriter(path, "group_size:1")
    for r in records[i * _SHARD_SIZE : (i + 1) * _SHARD_SIZE]:
      w.write(r)
    w.close()
    paths.append(path)
  return paths


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--n_train", type=int, default=2000)
  ap.add_argument("--n_val", type=int, default=256)
  ap.add_argument("--n_toy", type=int, default=10)
  args = ap.parse_args()

  out = schema.LOCAL_DATA_ROOT
  os.makedirs(out, exist_ok=True)
  manifest = {}

  print(f"converting train ({args.n_train}, human-only) ...")
  train = list(iter_hf("train", args.n_train))
  manifest["train_paths"] = write_arrayrecord_shards(
      train, out, schema.TRAIN_PATTERN
  )
  manifest["n_train"] = len(train)

  print(f"converting val ({args.n_val}) ...")
  val = list(iter_hf("val", args.n_val))
  manifest["val_paths"] = write_arrayrecord_shards(val, out, schema.VAL_PATTERN)
  manifest["n_val"] = len(val)

  print(f"converting toy ({args.n_toy}, Bagz) ...")
  toy = val[: args.n_toy]
  toy_path = os.path.join(out, schema.TOY_BAGZ)
  with bagz.Writer(toy_path) as w:
    for r in toy:
      w.write(r)
  manifest["toy_path"] = toy_path
  manifest["n_toy"] = len(toy)
  # Fixed-square preprocessing makes n_soft(256) < budget(280) for EVERY image
  # -> padded patch rows exist for every toy example (C6c precondition).
  manifest["padding_note"] = (
      "fixed-square preprocessing: actual n_soft=256 < budget 280 =>"
      " padded patch rows exist for every example"
  )
  manifest["digest"] = hashlib.sha256(b"".join(train[:8] + val[:8])).hexdigest()

  mpath = os.path.join(out, "dataset_manifest.json")
  with open(mpath, "w") as f:
    json.dump(manifest, f, indent=2)
  print(f"DONE: {manifest['n_train']} train / {manifest['n_val']} val /"
        f" {manifest['n_toy']} toy -> {out}")


if __name__ == "__main__":
  main()
