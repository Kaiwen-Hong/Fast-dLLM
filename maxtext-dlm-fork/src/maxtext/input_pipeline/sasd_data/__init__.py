# VENDORED from Fast-dLLM jax_ddrive @ b18e861 (branch jax-ddrive-port).
# Do NOT edit here — edit the source repo and re-copy (see PATCHES.md "sasd_data vendor").
"""Self-contained SASD data path for dataset_type='waymo_sasd' (no ddrive_jax needed).

Modules: schema (dtype contract), parquet_dataset (v1 eager source), ar_dataset
(v2 ArrayRecord lazy source, optional precomputed image_embeds), noise (online SASD
noising), grain_pipeline (multi-host grain MapDataset loader — the public entry is
``make_sasd_loader``).
"""
from maxtext.input_pipeline.sasd_data.grain_pipeline import SasdLoader, make_sasd_loader  # noqa: F401
