"""T0 — gs:// reachability + ckpt tree dump + tokenizer-id assertion (CPU only).

Answers, without downloading full checkpoints:
  1. Are GEMMA4_E2B_{PT,IT} reachable from this PC?
  2. Does each ckpt contain a vision tower (and audio)? Param counts per subtree.
  3. Do local Gemma4Tokenizer special ids match the internal table
     (105/106/258880/255999/258882; pipeline "\n\n"=108)?
Emits /home/kaiwen/data/dgemma_e2b/feas/t0_gs_probe.json.
"""

import json
import os
import sys
import traceback

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2b_feas_common as common  # noqa: E402

OUT = {}
CKPTS = {
    "pt": "gs://gemma-data/checkpoints/gemma4-e2b-pt",
    "it": "gs://gemma-data/checkpoints/gemma4-e2b-it",
}
TOKENIZER_GS = "gs://gemma-data/tokenizers/tokenizer_gemma4.model"

# Internal table [internal·reply, response-from-jestki.md Doc 2 §5]
INTERNAL_IDS = {
    "PAD": 0, "EOS": 1, "BOS": 2,
    "START_OF_TURN": 105, "END_OF_TURN": 106,
    "IMAGE_PLACEHOLDER": 258880, "START_OF_IMAGE": 255999, "END_OF_IMAGE": 258882,
}

from etils import epath  # noqa: E402

# ---- 1) reachability + listing ----
for name, path in list(CKPTS.items()) + [("tokenizer", TOKENIZER_GS)]:
  try:
    p = epath.Path(path)
    exists = p.exists()
    entry = {"exists": bool(exists)}
    if exists and name != "tokenizer":
      entry["children"] = sorted(c.name for c in p.iterdir())[:20]
    OUT[f"gs_{name}"] = entry
    print(f"T0: gs {name}: exists={exists} {entry.get('children', '')}")
  except Exception as e:  # noqa: BLE001
    OUT[f"gs_{name}"] = {"error": repr(e)}
    print(f"T0: gs {name}: ERROR {e!r}")

# ---- 2) ckpt tree metadata (no full download) ----
try:
  import orbax.checkpoint as ocp
  import orbax.checkpoint._src.path.step as _step_lib
  _step_lib.is_path_finalized = lambda path: True  # same patch as gemma_checkpointer.py

  for name, path in CKPTS.items():
    if not OUT.get(f"gs_{name}", {}).get("exists"):
      continue
    tree = None
    for candidate in [path] + [
        f"{path}/{c}" for c in OUT[f"gs_{name}"].get("children", [])
    ]:
      try:
        meta = ocp.PyTreeCheckpointer().metadata(candidate)
        tree = meta.item_metadata.tree if hasattr(meta, "item_metadata") else meta.tree
        OUT[f"tree_{name}_path"] = candidate
        break
      except Exception:  # noqa: BLE001
        continue
    if tree is None:
      OUT[f"tree_{name}"] = {"error": "no readable orbax tree at root or children"}
      print(f"T0: tree {name}: UNREADABLE")
      continue
    counts = common.subtree_param_counts(tree)
    top = {k: v for k, v in counts.items() if k != "__total__"}
    flatkeys = " ".join(sorted(top))
    has_vision = any("vision" in k.lower() for k in top)
    has_audio = any("audio" in k.lower() for k in top)
    has_sc = any("self_conditioner" in k.lower() for k in top)
    OUT[f"tree_{name}"] = {
        "total_params": counts["__total__"],
        "subtrees": top,
        "has_vision_tower": has_vision,
        "has_audio_tower": has_audio,
        "has_self_conditioner": has_sc,
    }
    print(
        f"T0: tree {name}: total={counts['__total__']/1e9:.2f}B"
        f" vision={has_vision} audio={has_audio} sc={has_sc} | subtrees: {flatkeys}"
    )
except Exception as e:  # noqa: BLE001
  OUT["tree_error"] = repr(e)
  traceback.print_exc()

# ---- 3) tokenizer ids: class-level + literal encodes ----
try:
  from gemma import gm
  st = gm.text.Gemma4Tokenizer.special_tokens
  local_ids = {k: int(getattr(st, k)) for k in INTERNAL_IDS if hasattr(st, k)}
  mismatches = {
      k: {"local": local_ids.get(k), "internal": v}
      for k, v in INTERNAL_IDS.items()
      if local_ids.get(k) != v
  }
  OUT["tokenizer_class_ids"] = local_ids
  OUT["tokenizer_id_mismatches"] = mismatches
  print(f"T0: tokenizer class ids: {local_ids}")
  print(f"T0: id mismatches vs internal: {mismatches or 'NONE — match'}")

  try:  # literal encodes need the vocab file (downloads from gs)
    tok = gm.text.Gemma4Tokenizer()
    literals = {}
    for s in ["<|image|>", "<|image", "<image|>", "<|turn>", "<turn|>", "\n\n"]:
      literals[repr(s)] = tok.encode(s)
    OUT["tokenizer_literal_encodes"] = {k: list(map(int, v)) for k, v in literals.items()}
    print(f"T0: literal encodes: {OUT['tokenizer_literal_encodes']}")
  except Exception as e:  # noqa: BLE001
    OUT["tokenizer_literal_encodes"] = {"error": repr(e)}
    print(f"T0: literal encode ERROR (vocab fetch?): {e!r}")
except Exception as e:  # noqa: BLE001
  OUT["tokenizer_error"] = repr(e)
  traceback.print_exc()

os.makedirs(common.RESULTS_DIR, exist_ok=True)
out_path = os.path.join(common.RESULTS_DIR, "t0_gs_probe.json")
with open(out_path, "w") as f:
  json.dump(OUT, f, indent=2, default=str)
print(f"T0: DONE -> {out_path}")
