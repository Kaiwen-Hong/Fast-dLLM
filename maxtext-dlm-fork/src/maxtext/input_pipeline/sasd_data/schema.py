# VENDORED from Fast-dLLM jax_ddrive @ b18e861 (branch jax-ddrive-port).
# Do NOT edit here — edit the source repo and re-copy (see PATCHES.md "sasd_data vendor").
"""SASD sample schema constants (mirror of ddrive_jax/convert/prep_to_parquet.py).

Single source of the per-array stored dtypes that drive both the parquet
``np.frombuffer`` reconstruction and the ArrayRecord ``tf.io.parse_tensor`` decode.
"""

ARRAY_DTYPES = {
    "input_ids": "int64", "labels": "int64", "rbi": "int32", "turn": "int32",
    "scaffold": "bool", "weight_vec": "float32", "block_alpha": "float32",
    "block_beta": "float32", "position_ids": "int32", "vision_mask": "bool",
    "pixel_values": "float16", "image_grid_thw": "int64",
}
ARRAY_FIELDS = list(ARRAY_DTYPES)
