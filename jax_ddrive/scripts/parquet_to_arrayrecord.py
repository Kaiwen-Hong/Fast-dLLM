"""Convert SASD parquet shards -> ArrayRecord shards (lossless, framework-free records).

Each ArrayRecord record = np.savez bytes of one frame's full dict (the 12 SASD arrays +
sample_id/L/n_blocks). Decode contract:
    import io, numpy as np
    d = np.load(io.BytesIO(record))          # -> dict-like of the arrays
ArrayRecord is grain-native (grain.ArrayRecordDataSource) for MaxText/TPU training.

  python parquet_to_arrayrecord.py <SRC_PARQUET_DIR> <DST_AR_DIR>
"""
import io, glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq
from array_record.python.array_record_module import ArrayRecordWriter

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.data.parquet_dataset import decode_row

SRC, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)
pfiles = sorted(glob.glob(os.path.join(SRC, "train-*.parquet")))
assert pfiles, f"no parquet under {SRC}"
nsh = len(pfiles)

total = 0
for i, pf in enumerate(pfiles):
    rows = pq.read_table(pf).to_pylist()
    out = os.path.join(DST, f"train-{i:05d}-of-{nsh:05d}.arrayrecord")
    w = ArrayRecordWriter(out, "group_size:1")
    for row in rows:
        r = decode_row(row)
        save = {k: v for k, v in r.items() if isinstance(v, np.ndarray)}
        save["sample_id"] = np.array(r["sample_id"])
        save["L"] = np.int64(r["L"])
        save["n_blocks"] = np.int64(r["n_blocks"])
        buf = io.BytesIO()
        np.savez(buf, **save)
        w.write(buf.getvalue())
    w.close()
    total += len(rows)
    print(f"  shard {i}: {len(rows)} records -> {os.path.basename(out)}", flush=True)

json.dump({"format": "arrayrecord", "split": "train", "num_samples": total, "num_shards": nsh,
           "record_encoding": "np.savez bytes per record",
           "decode": "import io,numpy as np; d=np.load(io.BytesIO(record))",
           "files": [f"train-{i:05d}-of-{nsh:05d}.arrayrecord" for i in range(nsh)]},
          open(os.path.join(DST, "dataset_info_train.json"), "w"), indent=2)
print(f"ARRAYRECORD_DONE total={total} shards={nsh}", flush=True)
