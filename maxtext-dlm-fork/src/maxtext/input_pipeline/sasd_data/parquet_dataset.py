# VENDORED from Fast-dLLM jax_ddrive @ b18e861 (branch jax-ddrive-port).
# Do NOT edit here — edit the source repo and re-copy (see PATCHES.md "sasd_data vendor").
"""Reconstruct Fast-dDrive SASD samples from the TPU-ready Parquet shards written by
``ddrive_jax/convert/prep_to_parquet.py``.

The decode contract is **bit-exact**: every array column is a little-endian binary blob,
reconstructed as ``np.frombuffer(blob, ARRAY_DTYPES[name]).reshape(<name>_shape)``. This is
the single source of truth the grain input pipeline (multi-host TPU) builds on, and is
deliberately framework-free (numpy only) so it runs identically on a TPU host, a GPU box,
or in CPU device-emulation.
"""
import glob
import os

import numpy as np

from maxtext.input_pipeline.sasd_data.schema import ARRAY_DTYPES, ARRAY_FIELDS


def decode_row(row: dict) -> dict:
    """One pyarrow row dict -> {sample_id, L, n_blocks, <arrays as np.ndarray>}. Writable copies."""
    out = {"sample_id": row["sample_id"], "L": int(row["L"]), "n_blocks": int(row["n_blocks"])}
    for name in ARRAY_FIELDS:
        buf = np.frombuffer(row[name], dtype=np.dtype(ARRAY_DTYPES[name]))
        out[name] = buf.reshape(tuple(int(x) for x in row[name + "_shape"])).copy()
    return out


def shard_paths(data_dir: str, split: str = "train") -> list:
    return sorted(glob.glob(os.path.join(data_dir, f"{split}-*.parquet")))


def iter_rows(paths):
    """Yield decoded sample dicts in shard order (host-side, numpy)."""
    import pyarrow.parquet as pq
    for p in paths:
        for row in pq.read_table(p).to_pylist():
            yield decode_row(row)


def load_all(data_dir: str, split: str = "train") -> list:
    """Eager-load a (small) split into a list of decoded sample dicts."""
    return list(iter_rows(shard_paths(data_dir, split)))
