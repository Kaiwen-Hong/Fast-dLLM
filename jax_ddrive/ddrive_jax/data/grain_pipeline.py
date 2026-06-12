"""Multi-host SASD input pipeline (grain ``MapDataset``) for Fast-dDrive JAX training.

Builds an **infinite, deterministic, resumable** stream of numpy batch dicts from either
source format (auto-detected by file extension in the data dir):

* ``*.arrayrecord`` — dataset v2 (``ar_dataset.ArRecordSource``): lazy random access, scales
  to the full 415k set; records may carry precomputed ``image_embeds`` [N_img, D] bf16
  (single copy), which is passed through into the batch as ``image_embeds`` [B, N_img, D]
  (consumer doubles it via ``concat([ie, ie], axis=1)`` instead of running the frozen ViT).
* ``*.parquet`` — v1 (``parquet_dataset.decode_row`` contract): eager load, small subsets.

Online SASD noising (``ddrive_jax/diffusion/noise.make_batch``) is applied per emitted
sample in both cases.

Six guarantees (all tested in ``tests/test_grain_pipeline.py``):

1. **Per-host sharding.**  One global stream ``G = source(range(Ntotal)).shuffle(seed).repeat()``
   is built identically on every host.  Host ``h`` of ``H`` consumes ``G.slice(slice(h, None, H))``
   = global stream positions ``{h, h+H, h+2H, ...}``.  Within global step ``t`` the ``H`` hosts
   collectively consume positions ``t*H .. t*H+H-1`` -- pairwise-disjoint, union == the full
   per-step block.  Over any whole number of steps the union across hosts is exactly a prefix
   of the shuffled-repeated global stream.  Sharding is applied **on the shuffled global stream**
   (not a per-host pre-slice) so every host's epoch shuffle is identical and the global index is
   monotone and unique.

2. **Epoch shuffle.**  ``.shuffle(seed)`` then ``.repeat()`` -- grain reseeds the permutation
   per epoch from ``(seed, epoch)`` internally.  Same ``seed`` => identical stream on every host
   and across runs.  ``shuffle=False`` => pure sequential repeated order.

3. **Online SASD noise.**  ``.map_with_index`` tags each sample with ``gidx`` = its monotone
   position in the shuffled-repeated global stream.  The per-sample rng is folded so re-deriving
   it from ``(seed, global_step, gidx)`` reproduces identical noise (``_fold_rng``):
   ``global_step = gidx // process_count``.  This is pure & stateless in ``gidx``.

4. **Resumability.**  ``SasdLoader.state()`` -> ``{"grain": {"next_index": N}}`` (JSON-able).
   ``set_state`` restores it on a fresh iterator; the continuation is bit-identical because grain
   replays ``next_index`` deterministically and noise is a pure function of the replayed ``gidx``.
   No rng counter / epoch counter to persist.  (``loader.grain_iterator`` exposes grain's native
   ``.save()/.load()`` for orbax/TPU.)

5. **Variable L.**  The current subset is uniform (L=1184); plain stacking.  The padding path
   (non-uniform L/N -> pad to ``max_length`` multiple of ``bd_size`` with MASK_ID/-100/-1, zero
   loss on padded tokens) is a clearly-guarded TODO (``_pad_sample``); the uniform path asserts
   uniformity when ``max_length is None``.

6. **Real grain.**  Uses ``grain.MapDataset`` natively (the TPU-native loader); the same code
   path is the TPU adapter -- no numpy-sampler fallback.

Run env (CPU 8-device emulation as multi-host proxy)::

    export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive
    export JAX_PLATFORMS=cpu
    export XLA_FLAGS="--xla_force_host_platform_device_count=8"
"""
from __future__ import annotations

import numpy as np
import grain
import pyarrow.parquet as pq

from ddrive_jax.data import ar_dataset, parquet_dataset
from ddrive_jax.diffusion import noise

# --- SSOT constants (do not re-hardcode MASK_ID; import from the noise module) -----------
MASK_ID = noise.MASK_ID   # 151665
LABEL_PAD = -100
RBI_PAD = -1


# ============================================================================ row source ==
class _RowSource:
    """grain ``RandomAccessDataSource`` (``__len__`` + ``__getitem__``) over Parquet shards.

    Global sample ordering for sharding is ``range(Ntotal)`` in ``shard_paths`` order
    (shards sorted by filename, rows in shard order) -- bit-exact reproducible.

    For the 400-row subset we eager-decode every row once at construction (simplest, reuses
    ``parquet_dataset.decode_row`` verbatim).  For the full-scale dataset, swap the eager list
    for a per-shard lazy ``pyarrow.Table`` cache keyed on ``(shard_idx, local_row)`` (see the
    inline note in ``__getitem__``) -- the public protocol is unchanged.
    """

    def __init__(self, paths):
        self._paths = list(paths)
        # eager decode: 400 small rows -> a flat Python list of decoded sample dicts.
        # full-scale alt (lazy): keep self._tables[shard] pyarrow Tables + a flat index
        #   self._index[i] = (shard, local_row); decode on demand in __getitem__.
        self._rows = []
        for p in self._paths:
            for row in pq.read_table(p).to_pylist():
                self._rows.append(parquet_dataset.decode_row(row))

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        # full-scale lazy variant:
        #   shard, r = self._index[i]
        #   return parquet_dataset.decode_row(self._tables[shard].slice(r, 1).to_pylist()[0])
        return self._rows[i]


# ===================================================================== deterministic rng ==
def _fold_rng(seed, global_step, global_sample_index):
    """Fold ``(seed, global_step, global_sample_index)`` -> a fresh numpy Generator.

    Uses ``SeedSequence(spawn_key=...)`` to combine the two coordinates without collision
    (numpy's intended high-quality seeding).  Both ``global_step`` and ``gidx`` are included so
    the "re-derive rng from step reproduces noise" property holds whether a harness keys on
    ``step`` or ``gidx`` (bijective for fixed ``process_count``).  Pure: identical inputs =>
    identical Generator => identical ``noise.make_batch`` output (resumability guarantee).
    """
    ss = np.random.SeedSequence(
        entropy=int(seed),
        spawn_key=(int(global_step), int(global_sample_index)),
    )
    return np.random.default_rng(ss)


# ==================================================================== per-sample transform ==
def _per_sample_arrays(s, ifn, lfn, ol, w, gidx, step):
    """Assemble the per-sample record (pre-batch) the batch dict needs."""
    rec = {
        "input_final": ifn,                                            # (2,2L) i64
        "labels_final": lfn,                                           # (2,L)  i64
        "original_labels": ol,                                         # (1,L)  i64
        "weights": w,                                                  # (2,L)  f32
        "position_ids": np.ascontiguousarray(s["position_ids"], np.int32),   # (3,L) i32
        "rbi": np.ascontiguousarray(s["rbi"], np.int32),              # (L,)  i32
        "turn": np.ascontiguousarray(s["turn"], np.int32),            # (L,)  i32
        "scaffold": np.ascontiguousarray(s["scaffold"], bool),        # (L,)  bool
        "pixel_values": np.ascontiguousarray(s["pixel_values"], np.float16),  # (N,1176) f16
        "image_grid_thw": np.ascontiguousarray(s["image_grid_thw"], np.int64),  # (n_img,3) i64
        "num_items": np.float32(noise.num_items(s)),                  # scalar f32
        "sample_id": s["sample_id"],                                  # str
        "_gidx": int(gidx),
        "_step": int(step),
    }
    if "image_embeds" in s:                                           # dataset v2 (precomputed
        rec["image_embeds"] = np.ascontiguousarray(s["image_embeds"])  # frozen-ViT) [N_img,D] bf16
    return rec


def _collate(records):
    """``batch_fn``: uniform path -> plain ``np.stack`` per array key.

    ``step`` = global training step of the first sample in the host-batch (monotone,
    == ``gidx0 // process_count``).  ``_gidx`` / ``_step`` are dropped from the public dict.
    """
    stack = lambda k: np.stack([r[k] for r in records], 0)
    out = {
        "input_final": stack("input_final"),          # (B,2,2L) i64
        "labels_final": stack("labels_final"),         # (B,2,L)  i64
        "original_labels": stack("original_labels"),   # (B,1,L)  i64
        "weights": stack("weights"),                   # (B,2,L)  f32
        "position_ids": stack("position_ids"),         # (B,3,L)  i32
        "rbi": stack("rbi"),                            # (B,L)    i32
        "turn": stack("turn"),                          # (B,L)    i32
        "scaffold": stack("scaffold"),                  # (B,L)    bool
        "pixel_values": stack("pixel_values"),          # (B,N,1176) f16
        "image_grid_thw": stack("image_grid_thw"),      # (B,n_img,3) i64
        "num_items": stack("num_items").astype(np.float32),  # (B,) f32
        "sample_id": [r["sample_id"] for r in records],      # list[str] len B
        "step": int(records[0]["_step"]),                    # global step (int)
    }
    if "image_embeds" in records[0]:
        out["image_embeds"] = stack("image_embeds")          # (B,N_img,D) bf16, single copy
    return out


# =================================================================== padding path (TODO) ==
def _pad_sample(s, max_length, max_N):
    """Variable-L/variable-N padding contract (TODO -- not used on the uniform subset).

    Right-pad the token axis to ``max_length`` (a multiple of ``bd_size``) and the pixel axis
    to ``max_N``::

        input_ids / noisy           -> MASK_ID (151665)
        labels / labels_final       -> -100
        original_labels             -> -100
        rbi                         -> -1
        turn                        -> -1
        scaffold                    -> True   (frozen: never noised, never loss'd)
        weight_vec / weights        -> 0.0
        position_ids                -> repeat last column (or 0; irrelevant since masked)
        pixel_values                -> 0.0,  to max N
        image_grid_thw              -> 0,    to max n_img

    **Loss-zero invariant:** padded token positions have ``labels_final == -100`` AND
    ``weights == 0.0``, so ``section_weighted_ce`` / ``causal_ce`` (which gate on
    ``labels != -100`` and divide by ``num_items``) contribute exactly zero.  ``num_items`` is
    computed from the **unpadded** labels (``noise.num_items(s)`` before padding) so the
    normaliser is unaffected: padded-batch loss == unpadded-batch loss for the same content.
    """
    raise NotImplementedError(
        "variable-L padding path: see _pad_sample docstring. "
        "The current subset is uniform; pass max_length=None for the stacking path."
    )


# ============================================================================== loader ====
class SasdLoader:
    """Infinite, resumable iterator of numpy batch dicts (see module docstring)."""

    def __init__(self, iter_dataset, per_host_batch, total_global_samples,
                 *, seed, process_index, process_count, L, N, n_img, has_embeds=False):
        self.has_embeds = bool(has_embeds)   # dataset v2: batches carry image_embeds
        self._ds = iter_dataset
        self.per_host_batch = per_host_batch
        self.total_global_samples = total_global_samples
        self.seed = seed
        self.process_index = process_index
        self.process_count = process_count
        self.L = L
        self.N = N
        self.n_img = n_img
        self._it = iter(self._ds)

    def __iter__(self):
        self._it = iter(self._ds)
        return self

    def __next__(self) -> dict:
        return next(self._it)

    # -- resume API (first-class) --------------------------------------------------------
    def state(self) -> dict:
        """JSON-able resume state: ``{"grain": {"next_index": N}}``."""
        return {"grain": self._it.get_state()}

    def set_state(self, st: dict) -> None:
        """Restore position; the next ``__next__`` continues identically."""
        self._it.set_state(st["grain"])

    @property
    def grain_iterator(self):
        """The underlying grain ``DatasetIterator`` (for native ``.save()/.load()`` on TPU)."""
        return self._it


def make_sasd_loader(parquet_dir, split="train", *, per_host_batch, seed=0,
                     process_index=0, process_count=1, shuffle=True, bd_size=32,
                     max_length=None, drop_remainder=True):
    """Build a :class:`SasdLoader` (infinite, deterministic, resumable).

    Args mirror the multi-host contract.  ``process_index`` / ``process_count`` default to a
    single host; pass ``jax.process_index()`` / ``jax.process_count()`` at the call site.
    """
    ar_paths = ar_dataset.shard_paths(parquet_dir, split)
    pq_paths = parquet_dataset.shard_paths(parquet_dir, split)
    if ar_paths and pq_paths:
        raise ValueError(f"both arrayrecord and parquet shards under {parquet_dir!r} — ambiguous")
    if ar_paths:
        src = ar_dataset.ArRecordSource(ar_paths)      # lazy random access (full-scale path)
    elif pq_paths:
        src = _RowSource(pq_paths)                     # eager decode (small subsets)
    else:
        raise FileNotFoundError(
            f"no {split}-*.arrayrecord or {split}-*.parquet shards under {parquet_dir!r}")
    ntotal = len(src)
    if ntotal == 0:
        raise ValueError(f"empty dataset under {parquet_dir!r} split={split!r}")

    # -- uniform vs padding path ---------------------------------------------------------
    # Probe a deterministic sample of records instead of scanning all rows (the lazy AR
    # source would otherwise force a full decode pass; both the 50k and 415k sets were
    # verified 100% uniform offline: L=1184, pixel (672,1176), 3 images).
    probe_idx = sorted(set(np.linspace(0, ntotal - 1, num=min(16, ntotal), dtype=int).tolist()))
    probe = [src[i] for i in probe_idx]
    Ls = sorted({int(r["input_ids"].shape[0]) for r in probe})
    Ns = sorted({int(r["pixel_values"].shape[0]) for r in probe})
    n_imgs = sorted({int(r["image_grid_thw"].shape[0]) for r in probe})
    has_embeds = {("image_embeds" in r) for r in probe}
    if len(has_embeds) != 1:
        raise ValueError("mixed shards: some records carry image_embeds, some do not")
    if max_length is None:
        assert len(Ls) == 1, (
            f"non-uniform L {Ls} requires max_length (padding path TODO; see _pad_sample)")
        assert len(Ns) == 1, (
            f"non-uniform pixel N {Ns} requires max_length (padding path TODO)")
        L = Ls[0]
        N = Ns[0]
        n_img = n_imgs[0]
    else:
        assert max_length % bd_size == 0, "max_length must be a multiple of bd_size"
        raise NotImplementedError(
            "variable-L padding path: see _pad_sample TODO. The current subset is uniform; "
            "call with max_length=None.")

    # -- grain MapDataset chain (sharding on the shuffled global stream) ------------------
    base = grain.MapDataset.source(src)
    g = base.shuffle(seed=int(seed)) if shuffle else base
    g = g.repeat()                                        # infinite
    g = g.map_with_index(lambda idx, x: (idx, x))         # monotone global_sample_index
    g = g.slice(slice(int(process_index), None, int(process_count)))  # PER-HOST partition

    pc = int(process_count)
    sd = int(seed)

    def _noise_then_arrays(record):
        gidx, s = record
        step = gidx // pc
        rng = _fold_rng(sd, step, gidx)
        ifn, lfn, ol, w = noise.make_batch(s, rng)
        return _per_sample_arrays(s, ifn, lfn, ol, w, gidx, step)

    g = g.map(_noise_then_arrays)
    g = g.batch(int(per_host_batch), drop_remainder=bool(drop_remainder), batch_fn=_collate)
    ds = g.to_iter_dataset(
        read_options=grain.ReadOptions(num_threads=8, prefetch_buffer_size=64))

    return SasdLoader(ds, int(per_host_batch), ntotal, seed=sd,
                      process_index=int(process_index), process_count=pc,
                      L=L, N=N, n_img=n_img, has_embeds=next(iter(has_embeds)))
