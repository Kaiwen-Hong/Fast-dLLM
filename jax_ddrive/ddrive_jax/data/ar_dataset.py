"""ArrayRecord(tf.train.Example) random-access source for Fast-dDrive SASD samples.

Reads the dataset-v2 shards written by ``jax_ddrive/scripts/parquet_to_ar_with_embeds.py``
(and the v1 shards from ``parquet_file_to_tfexample_ar.py`` — ``image_embeds`` is optional).
Decode contract mirrors ``parquet_dataset.decode_row`` exactly: same keys, same dtypes,
same shapes — plus, when present, ``image_embeds`` [N_img_tokens, D] bfloat16 (SINGLE copy;
the consumer doubles it via ``concat([ie, ie], 0)`` for the [noisy|clean] sequence).

TF is required only to parse ``tf.train.Example``/``parse_tensor`` (it is already a MaxText
dependency).  It is imported lazily and pinned to CPU so it never touches the accelerator
owned by JAX in the training process.

``ArRecordSource`` satisfies grain's RandomAccessDataSource protocol (``__len__`` +
``__getitem__``) and is what ``grain_pipeline.make_sasd_loader`` builds on for
``*.arrayrecord`` data dirs — random access, so the full 415k set needs no eager load.
"""
from __future__ import annotations

import glob
import os

import numpy as np

_TF = None


def _tf():
    """Lazy TF import, pinned to CPU (best effort if devices already initialized)."""
    global _TF
    if _TF is None:
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass  # devices already initialized elsewhere; parse ops are CPU-placed anyway
        _TF = tf
    return _TF


# Field -> TF parse dtype. Mirrors prep_to_parquet.ARRAY_DTYPES (+ optional image_embeds).
def _ar_dtypes(tf):
    return {
        "input_ids": tf.int64, "labels": tf.int64, "rbi": tf.int32, "turn": tf.int32,
        "scaffold": tf.bool, "weight_vec": tf.float32, "block_alpha": tf.float32,
        "block_beta": tf.float32, "position_ids": tf.int32, "vision_mask": tf.bool,
        "pixel_values": tf.half, "image_grid_thw": tf.int64,
        "image_embeds": tf.bfloat16,
    }


def decode_example(record_bytes: bytes) -> dict:
    """One serialized tf.train.Example -> sample dict (numpy, writable).

    Returns the same dict as ``parquet_dataset.decode_row`` plus ``image_embeds``
    when the record carries it (dataset v2).
    """
    tf = _tf()
    ex = tf.train.Example.FromString(record_bytes)
    f = ex.features.feature
    out = {
        "sample_id": f["sample_id"].bytes_list.value[0].decode(),
        "L": int(f["L"].int64_list.value[0]),
        "n_blocks": int(f["n_blocks"].int64_list.value[0]),
    }
    for name, dt in _ar_dtypes(tf).items():
        if name == "image_embeds" and name not in f:
            continue  # v1 shard (pixels only)
        out[name] = tf.io.parse_tensor(f[name].bytes_list.value[0], dt).numpy()
    return out


def shard_paths(data_dir: str, split: str = "train") -> list:
    return sorted(glob.glob(os.path.join(data_dir, f"{split}-*.arrayrecord")))


class ArRecordSource:
    """grain RandomAccessDataSource over ArrayRecord shards (lazy, thread-safe)."""

    def __init__(self, paths):
        from array_record.python.array_record_data_source import ArrayRecordDataSource
        self._paths = list(paths)
        self._ds = ArrayRecordDataSource(self._paths)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        return decode_example(self._ds[i])
