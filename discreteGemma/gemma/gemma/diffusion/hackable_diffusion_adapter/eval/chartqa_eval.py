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


# ---- Internal-parity scoring (verbatim from the internal SFT stack — see ----
# ---- discreteGemma/response-from-jestki.md Doc 2 Part 2 §3a).            ----


def extract_chartqa_answer(text: str) -> str:
  """Extract the final ChartQA answer string from generated text."""
  if text.endswith("<turn|>"):
    text = text[: -len("<turn|>")]
  match = re.search(r"(?i)the\s+answer\s+is[:\s]*(.*)", text)
  if match:
    return match.group(1).strip().rstrip(".,")
  return text.strip().rstrip(".,")


def _clean_numeric(text: str) -> str:
  # Remove spaces, commas, percentages, and dollar signs
  text = text.replace(" ", "").replace(",", "")
  text = text.rstrip("%").lstrip("$")
  return text


def score_chartqa(generated: str, ground_truth: str) -> bool:
  """Check whether hypothesis matches gt_answer.

  Applies an exact match for strings and a 5.1% tolerance relaxed match
  for numerical answers (following standard ChartQA protocol).
  """
  gen_ans = extract_chartqa_answer(generated)
  gt_ans = ground_truth.strip().rstrip(".,")

  if not gen_ans or not gt_ans:
    return False

  # Clean strings
  gen_clean = gen_ans.lower()
  gt_clean = gt_ans.lower()

  if gen_clean == gt_clean:
    return True

  # Try relaxed numeric match (5.1% tolerance)
  try:
    gen_num = float(_clean_numeric(gen_clean))
    gt_num = float(_clean_numeric(gt_clean))
    return abs(gen_num - gt_num) <= 0.051 * abs(gt_num)
  except ValueError:
    return False


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
  """ChartQA relaxed accuracy — internal-parity `score_chartqa`.

  Exact string match, else 5.1% relaxed numeric tolerance; the answer is
  extracted from the generation via the "The answer is:" prefix (which the
  data pipeline's `response_text` template guarantees).
  """

  def score_example(self, generated_text: str, ground_truth_text: str) -> bool:
    return score_chartqa(generated_text, ground_truth_text)


@dataclasses.dataclass(kw_only=True, frozen=True)
class ChartQAExactMatch(_LocalDecodeMetric):
  """Normalized exact-match accuracy (stricter companion metric)."""

  def score_example(self, generated_text: str, ground_truth_text: str) -> bool:
    return _normalize_text(generated_text) == _normalize_text(ground_truth_text)
