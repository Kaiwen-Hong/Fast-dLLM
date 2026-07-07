"""MM alignment property test (design doc B3.2) — CPU only.

Verifies the internal-parity strip+gather machinery:
  1. `get_no_mm_indices` collapses the expanded image block
     [\n\n, SOI, -2 x n, EOI, \n\n] to a single \n\n in the no-MM view.
  2. `remove_mm_logits` (pre-expanded flow) gathers logits with the SAME
     mapping (positions encode-identity check).
  3. `shift_encoder_targets_for_multimodal` gathers targets with the SAME
     mapping: gathered_target[j] == expanded_target[map(j)].
  4. Image at start / middle / end of prompt; a no-image row maps j->j.
"""

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma")

import jax.numpy as jnp
import numpy as np
from gemma.gm.vision import _token_utils
from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model

NL, SOI, EOI, PH = 108, 255999, 258882, -2
N_SOFT = 4
BLOCK = [NL, SOI] + [PH] * N_SOFT + [EOI, NL]  # len n+4; net insert vs <|image|> = n+3


def expanded_prompt(pos: str, plen: int = 20):
  """Build an expanded prompt with the image block at start/middle/end."""
  text = list(range(1000, 1000 + plen - len(BLOCK)))
  if pos == "start":
    row = BLOCK + text
  elif pos == "end":
    row = text + BLOCK
  else:
    k = len(text) // 2
    row = text[:k] + BLOCK + text[k:]
  return row


def run_case(pos: str):
  plen, clen = 20, 6
  prompt = np.array([expanded_prompt(pos, plen)], np.int32)
  canvas = np.arange(2000, 2000 + clen, dtype=np.int32)[None, :]
  full = np.concatenate([prompt, canvas], axis=1)  # [1, 26]
  L = full.shape[1]
  l_no_mm = L - (N_SOFT + 3)

  idx = np.asarray(_token_utils.get_no_mm_indices(
      tokens=jnp.asarray(full), l_no_mm=l_no_mm, num_tokens_per_image=N_SOFT))
  view = np.take_along_axis(full, idx, axis=1)[0]

  # 1) no-MM view: no PH/SOI/EOI; exactly one NL where the block was.
  assert PH not in view and SOI not in view and EOI not in view, (pos, view)
  assert list(view).count(NL) == 1, (pos, view)
  # text+canvas tokens all preserved in order:
  keep = [t for t in full[0] if t not in (PH, SOI, EOI, NL)]
  got = [t for t in view if t != NL]
  assert got == keep, (pos, got, keep)

  # 2) remove_mm_logits uses the SAME mapping (identity-encoded logits).
  V = 3
  logits = np.zeros((1, L, V), np.float32)
  logits[0, :, 0] = np.arange(L)
  stripped = np.asarray(_token_utils.remove_mm_logits(
      logits=jnp.asarray(logits), tokens=jnp.asarray(full),
      num_tokens_per_image=N_SOFT))
  assert stripped.shape == (1, l_no_mm, V)
  assert np.array_equal(stripped[0, :, 0].astype(int), idx[0]), pos

  # 3) target gather consistency: gathered[j] == expanded_target[map(j)].
  enc_target = np.roll(full, -1, axis=1)  # fake next-token targets
  enc_mask = np.ones_like(enc_target, np.float32)
  st, sm = sft_model.shift_encoder_targets_for_multimodal(
      encoder_target=jnp.asarray(enc_target),
      encoder_target_mask=jnp.asarray(enc_mask),
      prompt=jnp.asarray(prompt), x0_tokens=jnp.asarray(canvas),
      l_no_mm=l_no_mm, num_tokens_per_image=N_SOFT)
  assert np.array_equal(np.asarray(st)[0], enc_target[0][idx[0]]), pos
  assert np.asarray(sm).shape == (1, l_no_mm)
  print(f"  case image@{pos}: OK (l_no_mm={l_no_mm})")


def run_no_image():
  full = np.arange(1, 27, dtype=np.int32)[None, :]  # no image markers
  L = full.shape[1]
  l_no_mm = L - (N_SOFT + 3)
  idx = np.asarray(_token_utils.get_no_mm_indices(
      tokens=jnp.asarray(full), l_no_mm=l_no_mm, num_tokens_per_image=N_SOFT))
  assert np.array_equal(idx[0], np.arange(l_no_mm)), idx  # j -> j (truncated tail)
  print(f"  case no-image: OK (maps j->j, tail truncated to {l_no_mm})")


if __name__ == "__main__":
  for p in ("start", "middle", "end"):
    run_case(p)
  run_no_image()
  print("MM_ALIGNMENT_TEST PASS")
