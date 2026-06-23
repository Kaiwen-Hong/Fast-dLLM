"""Job ② loader wiring (doc 09 §4.4): `grain.ArrayRecordDataSource` -> `TokenizeSASD`.

This is the train-input side: a Grain data source over the msgpack ArrayRecord (Job ①
output) feeding the `TokenizeSASD` MapTransform, so each TPU-host worker decodes + tokenizes
+ builds SASD structure on the fly. Replaces the retired `ar_dataset.decode_example`
(tf.train.Example / tf.io.parse_tensor) path entirely.

Also exposes `read_raw_records` — a plain ArrayRecordReader helper (no grain) used by the
parity gate's round-trip check and by anyone who just wants the raw msgpack bytes back.

Local note: batching is NOT applied here. The 13 fields have per-sample-variable shapes
(L, n_blocks, N all vary), so real training appends a Pad+Batch stage after this transform;
that padding is training-config, not data, and is out of scope for this loader skeleton.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_raw_records(ar_path):
    """Return the list of raw msgpack `bytes` rows from an ArrayRecord file (no grain)."""
    from array_record.python.array_record_module import ArrayRecordReader
    reader = ArrayRecordReader(ar_path)
    n = reader.num_records()
    blobs = reader.read_all()
    assert len(blobs) == n, f"read_all returned {len(blobs)} != num_records {n}"
    return list(blobs)


def build_grain_loader(ar_path, *, section_w=None, noise_sched=None,
                       worker_count=0, shuffle=False, num_epochs=1, seed=0):
    """Wire ArrayRecordDataSource -> TokenizeSASD into a `grain.DataLoader` that yields the
    13-field per-sample dict. `worker_count=0` runs in-process (the Qwen processor is heavy
    to pickle across workers; bump it on the TPU host where prefetch overlap matters)."""
    import grain.python as grain
    from tokenize_sasd import TokenizeSASD

    source = grain.ArrayRecordDataSource(ar_path)
    sampler = grain.IndexSampler(
        num_records=len(source),
        shard_options=grain.NoSharding(),
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=seed,
    )
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[TokenizeSASD(section_w=section_w, noise_sched=noise_sched)],
        worker_count=worker_count,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test the Grain loader over an ArrayRecord file.")
    ap.add_argument("--ar_path", required=True)
    args = ap.parse_args()

    raw = read_raw_records(args.ar_path)
    print(f"[grain-loader] {len(raw)} raw records, sizes={[len(b) for b in raw]}")
    loader = build_grain_loader(args.ar_path)
    for i, sample in enumerate(loader):
        shapes = {k: getattr(v, "shape", ()) for k, v in sample.items()}
        print(f"[grain-loader] sample {i}: input_ids{shapes['input_ids']} "
              f"n_blocks={int(sample['n_blocks'])} pixels{shapes['pixel_values']}")
    print("GRAIN_LOADER_SMOKE_DONE")
