# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ChartQA evaluation metrics (relaxed accuracy + exact match).

Mirrors ``pubmedqa_eval`` / ``sudoku_eval``: a ``BaseSimpleTextMetric`` that
detokenizes the sampled answer (``samples``) and the ground-truth answer
(``batch.answer_tokens``) and scores each example.  Decoding uses the LOCAL
``gm.text.Gemma4Tokenizer`` (not the GCS seqio tokenizer of the base class), so
evaluation runs fully offline.

ChartQA's standard metric is *relaxed correctness*: for numeric answers the
prediction is correct if it is within 5% relative tolerance of the ground truth;
otherwise a normalized exact string match is required.
"""

from __future__ import annotations

import dataclasses
import functools
import re

from gemma import gm
from gemma.diffusion.hackable_diffusion_adapter.eval import base_metric
import numpy as np

_PAD = int(gm.text.Gemma4Tokenizer.special_tokens.PAD)
_EOS = int(gm.text.Gemma4Tokenizer.special_tokens.EOS)


def _normalize_text(s: str) -> str:
  """Lowercase, collapse whitespace, drop surrounding punctuation/%. """
  s = s.strip().lower()
  s = re.sub(r"\s+", " ", s)
  return s.strip(" .%$")


def _to_number(s: str) -> float | None:
  """Parse a ChartQA-style numeric answer (handles %, commas, $)."""
  s = s.strip().rstrip("%").replace(",", "").replace("$", "").strip()
  try:
    return float(s)
  except ValueError:
    return None


def relaxed_correctness(
    generated: str, ground_truth: str, tol: float = 0.05
) -> bool:
  """ChartQA relaxed correctness: 5% tolerance for numbers, else exact match."""
  gen, gt = generated.strip(), ground_truth.strip()
  gn, tn = _to_number(gen), _to_number(gt)
  if gn is not None and tn is not None:
    if tn == 0.0:
      return abs(gn) <= 1e-6
    return abs(gn - tn) / abs(tn) <= tol
  return _normalize_text(gen) == _normalize_text(gt)


@dataclasses.dataclass(kw_only=True, frozen=True)
class _LocalDecodeMetric(base_metric.BaseSimpleTextMetric):
  """BaseSimpleTextMetric that decodes with the local Gemma4 tokenizer."""

  @functools.cached_property
  def _local_tokenizer(self):
    return gm.text.Gemma4Tokenizer()

  def decode_batch(self, tokens_np: np.ndarray) -> list[str]:
    texts = []
    for row in tokens_np:
      ids = [int(t) for t in row.tolist()]
      # Truncate at the first EOS, drop PAD / negative placeholders.
      if _EOS in ids:
        ids = ids[: ids.index(_EOS)]
      ids = [t for t in ids if t != _PAD and t >= 0]
      texts.append(self._local_tokenizer.decode(ids).strip() if ids else "")
    return texts


@dataclasses.dataclass(kw_only=True, frozen=True)
class ChartQARelaxedAccuracy(_LocalDecodeMetric):
  """ChartQA relaxed accuracy (numeric within 5%, else normalized exact)."""

  tolerance: float = 0.05

  def score_example(self, generated_text: str, ground_truth_text: str) -> bool:
    return relaxed_correctness(
        generated_text, ground_truth_text, tol=self.tolerance
    )


@dataclasses.dataclass(kw_only=True, frozen=True)
class ChartQAExactMatch(_LocalDecodeMetric):
  """Normalized exact-match accuracy (stricter companion metric)."""

  def score_example(self, generated_text: str, ground_truth_text: str) -> bool:
    return _normalize_text(generated_text) == _normalize_text(ground_truth_text)
