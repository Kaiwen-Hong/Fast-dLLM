"""Convert SASD parquet shards -> ArrayRecord shards where each record is a serialized
`tf.train.Example` (the TFDS/grain-standard payload).

Each array field is stored as a bytes feature via `tf.io.serialize_tensor` (dtype+shape
preserved, incl. float16). Scalars: sample_id (bytes), L / n_blocks (int64).

Decode (TF):
    ex = tf.train.Example.FromString(record)
    input_ids = tf.io.parse_tensor(ex.features.feature['input_ids'].bytes_list.value[0], tf.int64)
    pixel     = tf.io.parse_tensor(ex.features.feature['pixel_values'].bytes_list.value[0], tf.half)
(dtype per field is in dataset_info_train.json -> array_dtypes)

  python parquet_to_tfexample_arrayrecord.py <SRC_PARQUET_DIR> <DST_AR_DIR>
"""
import glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
from array_record.python.array_record_module import ArrayRecordWriter

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.data.parquet_dataset import decode_row

SRC, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)

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


pfiles = sorted(glob.glob(os.path.join(SRC, "train-*.parquet")))
assert pfiles, f"no parquet under {SRC}"
nsh = len(pfiles)
total, dtypes = 0, {}
for i, pf in enumerate(pfiles):
    rows = pq.read_table(pf).to_pylist()
    out = os.path.join(DST, f"train-{i:05d}-of-{nsh:05d}.arrayrecord")
    w = ArrayRecordWriter(out, "group_size:1")
    for row in rows:
        r = decode_row(row)
        if not dtypes:
            dtypes = {k: str(r[k].dtype) for k in ARRAY_FIELDS}
        w.write(to_example(r).SerializeToString())
    w.close()
    total += len(rows)
    print(f"  shard {i}: {len(rows)} records -> {os.path.basename(out)}", flush=True)

json.dump({"format": "arrayrecord/tf.train.Example", "split": "train",
           "num_samples": total, "num_shards": nsh,
           "record_encoding": "tf.train.Example; array fields = tf.io.serialize_tensor bytes; "
                              "decode via tf.io.parse_tensor(bytes, dtype)",
           "array_fields": ARRAY_FIELDS, "array_dtypes": dtypes,
           "scalar_fields": {"sample_id": "bytes", "L": "int64", "n_blocks": "int64"},
           "files": [f"train-{i:05d}-of-{nsh:05d}.arrayrecord" for i in range(nsh)]},
          open(os.path.join(DST, "dataset_info_train.json"), "w"), indent=2)
print(f"TFEXAMPLE_AR_DONE total={total} shards={nsh}", flush=True)
