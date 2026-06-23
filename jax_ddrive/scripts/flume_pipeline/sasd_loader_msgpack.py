"""The NEW data loader: msgpack ArrayRecord -> TokenizeSASD -> the SAME SASD batch dict.

This is the train-input realization of Job ② (doc 09 §4.4): a grain source over the
flume_pipeline msgpack ArrayRecord whose `__getitem__` runs `TokenizeSASD` to produce the
prep-level row dict, then the EXACT downstream from `ddrive_jax.data.grain_pipeline`
(`_fold_rng` -> `noise.make_batch` -> `_per_sample_arrays` -> `_collate`) does the online SASD
doubling/noising/collation. Because the per-sample row dict is bit-identical to the prep npz
(proven by parity_test.py) AND the noising/collation are the *same imported functions* the
validated `make_sasd_loader` uses, the emitted batch dict is identical to the production loader's
for the same samples — so it drops straight into `train_tpu.run_step` with no other change.

This replaces the retired `ar_dataset.decode_example` (tf.train.Example) front-end with
`msgpack-unpack + TokenizeSASD`. Runs in the **ddrive** env (TokenizeSASD needs torch +
transformers 4.57.1). For tiny validation sets the source pre-materializes all rows in
`__init__` (thread-safe + fast); the google3 path would use per-worker processes instead.
"""
import os
import sys

import numpy as np
import grain

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("FASTDDRIVE_REPO",
                                  "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive"))

# Reuse the VALIDATED downstream verbatim (single source of truth -> identical batches).
from ddrive_jax.data.grain_pipeline import _fold_rng, _per_sample_arrays, _collate, SasdLoader  # noqa: E402
from ddrive_jax.diffusion import noise                                                          # noqa: E402

from tokenize_sasd import TokenizeSASD                                                           # noqa: E402
from etl_record import unpack                                                                    # noqa: E402


class MsgpackArSource:
    """grain RandomAccessDataSource over a flume_pipeline msgpack ArrayRecord.

    `__getitem__(i)` returns the prep-level row dict (the same contract as
    `parquet_dataset.decode_row`): {sample_id, L, n_blocks, + the 12 array fields}, which
    `noise.make_batch` / `_per_sample_arrays` consume downstream.
    """

    def __init__(self, ar_path, *, section_w=None, noise_sched=None, model_dir=None):
        from array_record.python.array_record_module import ArrayRecordReader
        reader = ArrayRecordReader(ar_path)
        blobs = list(reader.read_all())
        tk = TokenizeSASD(**({"model_dir": model_dir} if model_dir else {}),
                          section_w=section_w, noise_sched=noise_sched)
        self._rows = []
        for blob in blobs:                              # pre-materialize (tiny sets; thread-safe)
            row = tk.map(blob)                          # 13-field prep tensors
            row["sample_id"] = unpack(blob)["sample_id"]
            row["L"] = int(row["input_ids"].shape[0])
            self._rows.append(row)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]


def make_msgpack_sasd_loader(ar_path, *, per_host_batch, seed=0, process_index=0,
                             process_count=1, shuffle=True, section_w=None, noise_sched=None,
                             model_dir=None):
    """Build a :class:`SasdLoader` over the msgpack ArrayRecord. Mirrors
    `grain_pipeline.make_sasd_loader` lines 290-314 exactly, swapping in `MsgpackArSource`."""
    src = MsgpackArSource(ar_path, section_w=section_w, noise_sched=noise_sched, model_dir=model_dir)
    ntotal = len(src)
    if ntotal == 0:
        raise ValueError(f"empty msgpack ArrayRecord: {ar_path!r}")

    # uniform-L probe (validation sets are uniform L=1856) -------------------------------
    probe = [src[i] for i in range(min(16, ntotal))]
    Ls = sorted({int(r["input_ids"].shape[0]) for r in probe})
    Ns = sorted({int(r["pixel_values"].shape[0]) for r in probe})
    n_imgs = sorted({int(r["image_grid_thw"].shape[0]) for r in probe})
    assert len(Ls) == 1, f"non-uniform L {Ls} (padding path not used in validation)"
    assert len(Ns) == 1, f"non-uniform pixel N {Ns}"
    L, N, n_img = Ls[0], Ns[0], n_imgs[0]

    # grain MapDataset chain (identical to make_sasd_loader) ------------------------------
    base = grain.MapDataset.source(src)
    g = base.shuffle(seed=int(seed)) if shuffle else base
    g = g.repeat()
    g = g.map_with_index(lambda idx, x: (idx, x))
    g = g.slice(slice(int(process_index), None, int(process_count)))
    pc, sd = int(process_count), int(seed)

    def _noise_then_arrays(record):
        gidx, s = record
        step = gidx // pc
        rng = _fold_rng(sd, step, gidx)
        ifn, lfn, ol, w = noise.make_batch(s, rng)
        return _per_sample_arrays(s, ifn, lfn, ol, w, gidx, step)

    g = g.map(_noise_then_arrays)
    g = g.batch(int(per_host_batch), drop_remainder=True, batch_fn=_collate)
    ds = g.to_iter_dataset(read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=8))

    return SasdLoader(ds, int(per_host_batch), ntotal, seed=sd,
                      process_index=int(process_index), process_count=pc,
                      L=L, N=N, n_img=n_img, has_embeds=False)
