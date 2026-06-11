"""Single parquet file -> single ArrayRecord shard of tf.train.Example records.
Same encoding as parquet_to_tfexample_arrayrecord.py (tf.io.serialize_tensor per array).
Used as the per-file worker for the parallel full-dataset conversion. TF forced to CPU.

  CUDA_VISIBLE_DEVICES= python parquet_file_to_tfexample_ar.py <PARQUET> <OUT.arrayrecord>
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # no GPU (avoid PTX JIT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
from array_record.python.array_record_module import ArrayRecordWriter

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.data.parquet_dataset import decode_row

PARQUET, OUT = sys.argv[1], sys.argv[2]
ARRAY_FIELDS = ["input_ids", "labels", "rbi", "turn", "scaffold", "weight_vec",
                "block_alpha", "block_beta", "position_ids", "vision_mask",
                "pixel_values", "image_grid_thw"]


def _bytes(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _int64(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))


def to_example(r):
    feat = {k: _bytes(tf.io.serialize_tensor(tf.constant(r[k])).numpy()) for k in ARRAY_FIELDS}
    feat["sample_id"] = _bytes(str(r["sample_id"]).encode())
    feat["L"] = _int64(r["L"])
    feat["n_blocks"] = _int64(r["n_blocks"])
    return tf.train.Example(features=tf.train.Features(feature=feat))


tmp = OUT + ".tmp"
w = ArrayRecordWriter(tmp, "group_size:1")
n = 0
pf = pq.ParquetFile(PARQUET)
for batch in pf.iter_batches(batch_size=64):               # stream -> bounded RAM
    for row in batch.to_pylist():
        w.write(to_example(decode_row(row)).SerializeToString())
        n += 1
w.close()
os.replace(tmp, OUT)                                       # atomic -> resume-safe
print(f"{os.path.basename(OUT)}: {n} records", flush=True)
