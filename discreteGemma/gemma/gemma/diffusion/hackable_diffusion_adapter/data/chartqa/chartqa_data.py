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

"""ChartQA (VQA) data source + transforms for the DiffusionGemma SFT harness.

This is the multimodal analogue of ``sudoku_data.py``.  It emits the exact
fields the SFT model (``SFTDiffusion``) consumes -- ``prompt``, ``canvas``,
``canvas_id``, ``canvas_mask``, ``encoder_target``, ``encoder_target_mask`` --
PLUS the multimodal inputs ``patches`` / ``positions_xy`` that
``SFTDiffusion`` packs into a ``PreprocessedVisionInput`` and merges at the
``-2`` (``TOKEN_PLACEHOLDER``) positions of the prompt.

Design notes (so grain can batch and so the Gemma4 vision merge is happy):
  * Every image is resized to a FIXED square (``image_size`` x ``image_size``),
    so each example produces a constant ``N_SOFT = (image_size // patch_size)**2``
    number of vision soft tokens -> fixed-shape ``patches`` /
    ``positions_xy`` arrays (no ragged batching).
  * Patchification is done in PURE NUMPY (a faithful re-implementation of
    ``gm/nn/gemma4/vision/_images.patchify``) so grain worker processes never
    import JAX.
  * The prompt is built manually as
    ``[BOS, START_OF_IMAGE, (-2) * N_SOFT, END_OF_IMAGE, <question ...>]`` padded
    to ``prompt_len``.  The count of ``-2`` MUST equal ``N_SOFT`` (==
    ``sum(soft_token_counts)``), or the vision merge misplaces soft tokens.
  * The encoder AR-loss mask excludes the image block (``-2`` / ``<soi>`` /
    ``<eoi>``) so we never train next-token prediction on the placeholders.

Two data sources are provided:
  * ``mode='chartqa'`` -> real HuggingFace ``ahmed-masry/ChartQA``.
  * ``mode='color'``  -> a synthetic *color-grounding* task (solid-colour image
    + fixed question ``"What is the color?"`` + colour-name answer).  The answer
    is determined ONLY by the image, so it lets us prove -- through the official
    harness -- that the vision encoder plays a causal role (see the validation
    script), which real ChartQA cannot at a tiny/random-init 5090 scale.
"""

from __future__ import annotations

import dataclasses
import functools
import io
from typing import Any

from gemma import gm
from gemma.diffusion.hackable_diffusion_adapter.data import data as adapter_data
from gemma.diffusion.hackable_diffusion_adapter.data.chartqa import schema
from gemma.gm.nn.gemma4.vision import _encoder as vision_encoder
from gemma.gm.vision import _token_utils
import grain.python as grain
import jax
from kauldron import kd
from kauldron import random
from kauldron.data.py import base as kd_base
import numpy as np
from PIL import Image

# ``-2``: not a real vocab id; marks where vision soft tokens are merged in.
TOKEN_PLACEHOLDER = vision_encoder.TOKEN_PLACEHOLDER

_CHARTQA_HF_NAME = "ahmed-masry/ChartQA"

# Synthetic color-grounding palette (name -> RGB).
_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("red", (220, 40, 40)),
    ("green", (40, 180, 40)),
    ("blue", (40, 40, 220)),
    ("yellow", (230, 210, 40)),
)
_COLOR_QUESTION = "What is the color?"


def soft_tokens_per_image(image_size: int, patch_size: int) -> int:
  """Number of vision soft tokens for a square image (pooling_kernel_size=1)."""
  side = image_size // patch_size
  return side * side


################################################################################
# MARK: Pure-numpy patchify (mirror of gm/nn/gemma4/vision/_images.patchify)
################################################################################


def _patchify_square(image_f32: np.ndarray, patch_size: int):
  """Patchify a square [S, S, 3] float image into ([L, P], [L, 2]).

  Matches ``_images.patchify`` exactly:
    * patches: einops ``(h p)(w q) c -> (h w)(p q c)`` (row-major over (y, x)),
      patch_dim = patch_size**2 * 3.
    * positions_xy: ``(x, y)`` per patch, x indexes width, y indexes height.
  """
  side = image_f32.shape[0]
  h = w = side // patch_size
  patches = (
      image_f32.reshape(h, patch_size, w, patch_size, 3)
      .transpose(0, 2, 1, 3, 4)  # -> (h, w, p, q, c)
      .reshape(h * w, patch_size * patch_size * 3)
  )
  xs, ys = np.meshgrid(np.arange(w), np.arange(h))  # indexing='xy' -> [h, w]
  positions_xy = np.stack([xs, ys], axis=-1).reshape(h * w, 2)
  return patches.astype(np.float32), positions_xy.astype(np.int32)


def _to_rgb_uint8(image: Any) -> np.ndarray:
  """Coerce a raw record image (PIL / bytes / ndarray) to uint8 RGB HxWx3."""
  if isinstance(image, (bytes, bytearray)):
    image = Image.open(io.BytesIO(image))
  if isinstance(image, Image.Image):
    return np.asarray(image.convert("RGB"), dtype=np.uint8)
  arr = np.asarray(image)
  if arr.ndim == 2:
    arr = np.stack([arr] * 3, axis=-1)
  elif arr.shape[-1] == 4:
    arr = arr[..., :3]
  return arr.astype(np.uint8)


################################################################################
# MARK: Data sources
################################################################################


class PicklableChartQAReader(grain.RandomAccessDataSource):
  """Picklable HuggingFace ChartQA reader (lazy load for multiprocessing)."""

  def __init__(self, split: str, num_examples: int | None):
    self.split = split
    self.num_examples = num_examples
    self._ds = None

  @property
  def ds(self):
    if self._ds is None:
      import datasets  # pylint: disable=g-import-not-at-top

      d = datasets.load_dataset(_CHARTQA_HF_NAME, split=self.split)
      if self.num_examples is not None:
        d = d.select(range(min(self.num_examples, len(d))))
      self._ds = d
    return self._ds

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, index):
    ex = self.ds[int(index)]
    # ChartQA HF columns: image (PIL), query (str), label (str), type.
    return {
        "image": ex["image"],
        "query": ex.get("query", ex.get("question", "")),
        "label": str(ex["label"]),
    }

  def __getstate__(self):
    state = self.__dict__.copy()
    state["_ds"] = None
    return state


class SyntheticColorReader(grain.RandomAccessDataSource):
  """Synthetic solid-color grounding task (answer depends ONLY on the image)."""

  def __init__(self, num_examples: int, image_size: int, seed: int = 0):
    self.num_examples = num_examples
    self.image_size = image_size
    self.seed = seed

  def __len__(self):
    return self.num_examples

  def __getitem__(self, index):
    index = int(index)
    # Deterministic, balanced color assignment + mild per-pixel noise so the
    # image is not a single constant (forces the encoder to actually read it).
    rng = np.random.RandomState(self.seed * 100003 + index)
    name, rgb = _COLORS[index % len(_COLORS)]
    base = np.empty((self.image_size, self.image_size, 3), dtype=np.int16)
    base[:] = np.asarray(rgb, dtype=np.int16)
    noise = rng.randint(-15, 16, size=base.shape)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    # Repeat the colour so the (short) answer canvas is FULLY image-determined
    # rather than dominated by EOS fill -- otherwise the diffusion loss is spent
    # on the trivial EOS tokens and the image->answer signal is starved.
    label = " ".join([name] * 4)
    return {"image": img, "query": _COLOR_QUESTION, "label": label}


################################################################################
# MARK: Transforms
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class BuildChartQAInputs(grain.MapTransform):
  """Image -> (patches, positions_xy); question/answer -> prompt/response.

  Produces (all fixed shape except ``response``, which ``CanvasChunker``
  fixes downstream):
    * patches: float32 [N_SOFT, patch_size**2 * 3]
    * positions_xy: int32 [N_SOFT, 2]
    * prompt: int32 [prompt_len]  (image block + question, pad to prompt_len)
    * response: int32 [variable]  (answer tokens)
  """

  tokenizer: Any
  prompt_len: int
  n_soft: int
  image_size: int
  patch_size: int
  query_max_len: int
  bos_token: int
  soi_token: int
  eoi_token: int
  pad_token: int
  # pooling_kernel_size == 1 -> fast pure-numpy fixed-square patchify (tiny,
  # validated). > 1 -> the release preprocessing (aspect-ratio resize + pooling);
  # a fixed square input still yields a fixed soft-token count so grain batches.
  pooling_kernel_size: int = 1
  max_soft_tokens: int = 0
  # >0 (eval): also emit fixed-length ground-truth `answer_tokens` for metrics.
  answer_len: int = 0

  def map(self, features):
    # --- image -> fixed square -> patches ---
    img = _to_rgb_uint8(features["image"])
    pil = Image.fromarray(img).resize(
        (self.image_size, self.image_size), resample=Image.BICUBIC
    )
    if self.pooling_kernel_size == 1:
      image_f32 = np.asarray(pil, dtype=np.float32) / 255.0
      patches, positions_xy = _patchify_square(image_f32, self.patch_size)
    else:
      # Release path (matches the 26B vision tower: pooling=3, output_length=280).
      from gemma.gm.nn.gemma4.vision import _preprocessing as _pp  # pylint: disable=g-import-not-at-top

      p, pos, _ = _pp.preprocess_and_patchify(
          [np.asarray(pil)],
          patch_size=self.patch_size,
          max_soft_tokens=self.max_soft_tokens,
          pooling_kernel_size=self.pooling_kernel_size,
      )
      patches = np.asarray(p[0], dtype=np.float32)
      positions_xy = np.asarray(pos[0], dtype=np.int32)
    features["patches"] = patches
    features["positions_xy"] = positions_xy

    # --- prompt: [BOS, <soi>, (-2)*N_SOFT, <eoi>, question...] pad to prompt_len ---
    q_ids = self.tokenizer.encode(str(features["query"]), add_bos=False)
    q_ids = q_ids[: self.query_max_len]
    head = (
        [self.bos_token, self.soi_token]
        + [TOKEN_PLACEHOLDER] * self.n_soft
        + [self.eoi_token]
    )
    prompt = head + q_ids
    prompt = prompt[: self.prompt_len]
    if len(prompt) < self.prompt_len:
      prompt = prompt + [self.pad_token] * (self.prompt_len - len(prompt))
    features["prompt"] = np.asarray(prompt, dtype=np.int32)

    # --- response: answer tokens (leading space, mirrors gm tokenization) ---
    resp = self.tokenizer.encode(" " + str(features["label"]), add_bos=False)
    features["response"] = np.asarray(resp, dtype=np.int32)

    # --- (eval) fixed-length ground-truth answer tokens for metrics ---
    if self.answer_len > 0:
      ans = resp[: self.answer_len]
      ans = ans + [self.pad_token] * (self.answer_len - len(ans))
      features["answer_tokens"] = np.asarray(ans, dtype=np.int32)
    return features


@dataclasses.dataclass(kw_only=True, frozen=True)
class ChartQAEncoderTargets(grain.MapTransform):
  """Shifted next-token targets over prompt+canvas, excluding the image block.

  Identical to ``adapter_data.SequenceTargetShift`` except that positions whose
  token is an image-block id (``-2`` / ``<soi>`` / ``<eoi>``) are marked
  invalid, so no encoder AR loss is computed on/around the vision placeholders.
  """

  in_prompt: str = "prompt"
  in_canvas: str = "canvas"
  in_canvas_mask: str = "canvas_mask"
  out_encoder_target: str = "encoder_target"
  out_encoder_target_mask: str = "encoder_target_mask"
  pad_token: int = 0
  invalid_ids: tuple[int, ...] = ()

  def map(self, features):
    prompt = np.asarray(features[self.in_prompt]).flatten()
    canvas = np.asarray(features[self.in_canvas]).flatten()
    canvas_mask = np.asarray(features[self.in_canvas_mask]).flatten()

    invalid = np.zeros(prompt.shape, dtype=np.bool_)
    for tid in self.invalid_ids:
      invalid |= prompt == tid
    prompt_valid = (prompt != self.pad_token) & (~invalid)
    canvas_valid = canvas_mask.astype(np.bool_)

    full_seq = np.concatenate([prompt, canvas])
    full_valid = np.concatenate([prompt_valid, canvas_valid])

    encoder_target = np.roll(full_seq, -1)
    encoder_target[-1] = self.pad_token
    shifted_valid = np.roll(full_valid, -1)
    shifted_valid[-1] = False
    # Require BOTH the current and next position to be valid.
    encoder_target_mask = full_valid & shifted_valid

    features[self.out_encoder_target] = encoder_target
    features[self.out_encoder_target_mask] = encoder_target_mask
    return features


################################################################################
# MARK: DataSourceBase
################################################################################


@dataclasses.dataclass(frozen=True)
class ChartQADataset(kd_base.DataSourceBase):
  """Kauldron data source for real ChartQA or the synthetic color task."""

  mode: str = "chartqa"  # "chartqa" | "color"
  split: str = "train"
  num_examples: int | None = None
  image_size: int = 64
  color_seed: int = 0

  @functools.cached_property
  def data_source(self) -> grain.RandomAccessDataSource:
    if self.mode == "color":
      n = self.num_examples if self.num_examples is not None else 1024
      return SyntheticColorReader(n, self.image_size, self.color_seed)
    return PicklableChartQAReader(self.split, self.num_examples)

  def ds_for_current_process(self, rng: random.PRNGKey) -> grain.MapDataset:
    ds = grain.MapDataset.source(self.data_source)
    ds = ds.seed(rng.as_seed())
    if self.shard_by_process:
      ds = ds[jax.process_index() :: jax.process_count()]
    if self.shuffle:
      ds = ds.shuffle(seed=rng.fold_in("shuffle").as_seed())
    return ds


################################################################################
# MARK: Internal-parity (V2) sources + transforms — tf.Example records
################################################################################


class ChartQAArrayRecordDataSource(grain.RandomAccessDataSource):
  """Picklable ArrayRecord source (internal-parity; raw tf.Example bytes)."""

  def __init__(self, paths: tuple[str, ...]):
    self.paths = tuple(paths)
    self._src = None

  @property
  def src(self):
    if self._src is None:
      self._src = grain.ArrayRecordDataSource(list(self.paths))
    return self._src

  def __len__(self):
    return len(self.src)

  def __getitem__(self, index):
    return self.src[int(index)]

  def __getstate__(self):
    state = self.__dict__.copy()
    state["_src"] = None
    return state


class ChartQABagzDataSource(grain.RandomAccessDataSource):
  """Picklable Bagz source (toy set; exercises the trajectory-side reader)."""

  def __init__(self, path: str):
    self.path = path
    self._reader = None

  @property
  def reader(self):
    if self._reader is None:
      import bagz  # pylint: disable=g-import-not-at-top

      self._reader = bagz.Reader(self.path)
    return self._reader

  def __len__(self):
    return len(self.reader)

  def __getitem__(self, index):
    return self.reader[int(index)]

  def __getstate__(self):
    state = self.__dict__.copy()
    state["_reader"] = None
    return state


@dataclasses.dataclass(kw_only=True, frozen=True)
class ParseChartQARecord(grain.MapTransform):
  """Parses tf.train.Example from record bytes (verbatim internal parse)."""

  def map(self, raw_bytes: bytes) -> dict[str, Any]:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top

    example = tf.train.Example.FromString(bytes(raw_bytes))
    feature = example.features.feature

    image_bytes = feature[schema.FEATURE_IMAGE].bytes_list.value[0]
    question = feature[schema.FEATURE_QUESTION].bytes_list.value[0].decode(
        "utf-8"
    )
    answer = feature[schema.FEATURE_ANSWER].bytes_list.value[0].decode("utf-8")

    return {
        "raw_image": image_bytes,
        "prompt": question,
        "prompt_text": question,
        "short_answer": answer,
        "short_answer_text": answer,
        "response_text": schema.CHARTQA_RESPONSE_TEMPLATE.format(answer=answer),
    }


@dataclasses.dataclass(kw_only=True, frozen=True)
class BuildChartQAInputsV2(grain.MapTransform):
  """Internal-parity input builder (template prompt + variable expansion).

  * image: ``raw_image`` bytes -> fixed square (``image_size``) -> the RELEASE
    preprocessing (``preprocess_and_patchify`` with the E2B tower params:
    patch 16 / pooling 3 / budget 280 -> actual n_soft = 256 with padded patch
    rows — the C6c padding precondition).
  * prompt: ``CHARTQA_PROMPT_TEMPLATE.format(text=question)`` tokenized
    (add_bos), then the single ``<|image|>`` expanded via the pristine
    ``add_variable_extra_tokens_for_images`` ([\\n\\n, <|image, -2 x n_soft,
    <image|>, \\n\\n]) and padded/truncated to ``prompt_len``.
  * response: ``response_text`` ("The answer is: {answer}") tokenized — the
    metric's extraction prefix, load-bearing.
  """

  tokenizer: Any
  prompt_len: int
  n_soft: int
  image_size: int
  patch_size: int
  pooling_kernel_size: int
  max_soft_tokens: int
  pad_token: int
  answer_len: int = 0  # >0 (eval): emit fixed-length GT `answer_tokens`.

  def map(self, features):
    # --- image -> fixed square -> release preprocessing ---
    img = _to_rgb_uint8(features["raw_image"])
    pil = Image.fromarray(img).resize(
        (self.image_size, self.image_size), resample=Image.BICUBIC
    )
    from gemma.gm.nn.gemma4.vision import _preprocessing as _pp  # pylint: disable=g-import-not-at-top

    p, pos, counts = _pp.preprocess_and_patchify(
        [np.asarray(pil)],
        patch_size=self.patch_size,
        max_soft_tokens=self.max_soft_tokens,
        pooling_kernel_size=self.pooling_kernel_size,
    )
    features["patches"] = np.asarray(p[0], dtype=np.float32)
    features["positions_xy"] = np.asarray(pos[0], dtype=np.int32)
    assert int(counts[0]) == self.n_soft, (
        f"actual soft count {counts[0]} != configured n_soft {self.n_soft}"
    )

    # --- prompt: template -> tokenize -> expand <|image|> -> pad to 512 ---
    text = schema.CHARTQA_PROMPT_TEMPLATE.format(text=str(features["prompt_text"]))
    ids = self.tokenizer.encode(text, add_bos=True)
    expanded = _token_utils.add_variable_extra_tokens_for_images(
        tokens=np.asarray([ids], dtype=np.int32),
        soft_token_counts=[self.n_soft],
    )[0].tolist()
    expanded = expanded[: self.prompt_len]
    if len(expanded) < self.prompt_len:
      expanded = expanded + [self.pad_token] * (self.prompt_len - len(expanded))
    features["prompt"] = np.asarray(expanded, dtype=np.int32)

    # --- response: "The answer is: {answer}" tokens (canvas content) ---
    resp = self.tokenizer.encode(str(features["response_text"]), add_bos=False)
    features["response"] = np.asarray(resp, dtype=np.int32)

    # --- (eval) fixed-length ground-truth SHORT answer for the metric ---
    if self.answer_len > 0:
      ans = self.tokenizer.encode(str(features["short_answer_text"]), add_bos=False)
      ans = ans[: self.answer_len]
      ans = ans + [self.pad_token] * (self.answer_len - len(ans))
      features["answer_tokens"] = np.asarray(ans, dtype=np.int32)
    return features


@dataclasses.dataclass(frozen=True)
class ChartQARecordsDataset(kd_base.DataSourceBase):
  """Kauldron data source over local ArrayRecord shards / a Bagz file."""

  paths: tuple[str, ...] = ()
  fmt: str = "arrayrecord"  # "arrayrecord" | "bagz"

  @functools.cached_property
  def data_source(self) -> grain.RandomAccessDataSource:
    if self.fmt == "bagz":
      assert len(self.paths) == 1
      return ChartQABagzDataSource(self.paths[0])
    return ChartQAArrayRecordDataSource(self.paths)

  def ds_for_current_process(self, rng: random.PRNGKey) -> grain.MapDataset:
    ds = grain.MapDataset.source(self.data_source)
    ds = ds.seed(rng.as_seed())
    if self.shard_by_process:
      ds = ds[jax.process_index() :: jax.process_count()]
    if self.shuffle:
      ds = ds.shuffle(seed=rng.fold_in("shuffle").as_seed())
    return ds


def make_chartqa_records_ds(
    *,
    training: bool,
    batch_size: int,
    paths: tuple[str, ...],
    fmt: str = "arrayrecord",
    prompt_len: int = schema.CHARTQA_PROMPT_LEN,
    num_canvases: int = schema.CHARTQA_NUM_CANVASES,
    canvas_size: int = schema.CHARTQA_CANVAS_SIZE,
    n_soft: int = schema.E2B_N_SOFT,
    image_size: int = schema.E2B_IMAGE_SIZE,
    patch_size: int = schema.E2B_PATCH_SIZE,
    pooling_kernel_size: int = schema.E2B_POOLING_KERNEL_SIZE,
    max_soft_tokens: int = schema.E2B_MAX_SOFT_TOKENS,
    answer_len: int = 32,
    num_workers: int = 0,
) -> ChartQARecordsDataset:
  """Internal-parity pipeline over tf.Example records (design doc §5 B4).

  Unlike ``make_chartqa_ds``, batch_size > 1 is supported: the vision merge
  reshapes the packed soft embeddings per batch row (internal batch fix in
  ``_merge_mm_embeddings``). Encoder targets use the PLAIN shift — the image
  block is handled downstream by ``shift_encoder_targets_for_multimodal``
  (strip+gather; positions collapse cleanly, no invalid-id masking needed).
  """
  tokenizer = gm.text.Gemma4Tokenizer()
  st = gm.text.Gemma4Tokenizer.special_tokens

  transforms = [
      ParseChartQARecord(),
      BuildChartQAInputsV2(
          tokenizer=tokenizer,
          prompt_len=prompt_len,
          n_soft=n_soft,
          image_size=image_size,
          patch_size=patch_size,
          pooling_kernel_size=pooling_kernel_size,
          max_soft_tokens=max_soft_tokens,
          pad_token=int(st.PAD),
          answer_len=0 if training else answer_len,
      ),
      adapter_data.CanvasChunker(
          in_response="response",
          out_canvas="canvas",
          out_canvas_id="canvas_id",
          out_canvas_mask="canvas_mask",
          num_canvases=num_canvases,
          canvas_size=canvas_size,
          eos_token=int(st.EOS),
          pad_token=int(st.PAD),
      ),
      # PLAIN next-token shift (internal parity — see docstring above).
      ChartQAEncoderTargets(pad_token=int(st.PAD), invalid_ids=()),
      kd.data.Rearrange(key="canvas", pattern="c -> c 1"),
  ]

  keep_fields = [
      "patches",
      "positions_xy",
      "prompt",
      "canvas",
      "canvas_id",
      "canvas_mask",
      "encoder_target",
      "encoder_target_mask",
  ]
  if not training:
    keep_fields.append("answer_tokens")
  transforms.append(kd.data.Elements(keep=keep_fields))

  return ChartQARecordsDataset(
      paths=tuple(paths),
      fmt=fmt,
      shuffle=training,
      num_epochs=None if training else 1,
      batch_size=batch_size,
      num_workers=num_workers,
      transforms=transforms,
  )


################################################################################
# MARK: Builder
################################################################################


def make_chartqa_ds(
    *,
    training: bool,
    batch_size: int,
    prompt_len: int,
    num_canvases: int,
    canvas_size: int,
    n_soft: int,
    image_size: int,
    patch_size: int,
    mode: str = "chartqa",
    split: str | None = None,
    num_examples: int | None = None,
    color_seed: int = 0,
    query_max_len: int = 32,
    pooling_kernel_size: int = 1,
    max_soft_tokens: int = 0,
    answer_len: int = 16,
    num_workers: int = 0,
) -> ChartQADataset:
  """Build the ChartQA (or color-grounding) pipeline for the SFT harness.

  Args:
    training: Whether this is the train (shuffle, infinite) or eval split.
    batch_size: Per-process batch size.  MUST be 1 for the multimodal path --
      the Gemma4 vision merge (``merge_flat_embeddings``) only reconciles a
      size-1 vision batch against the text batch, so the model handles a single
      sequence per forward.
    prompt_len: Padded prompt length (image block + question).
    num_canvases: Number of canvas chunks (1 for short answers).
    canvas_size: Tokens per canvas chunk.
    n_soft: Vision soft tokens per image (== ``soft_tokens_per_image``).
    image_size: Square resize side (px).
    patch_size: Vision patch size (px).
    mode: ``"chartqa"`` (HF) or ``"color"`` (synthetic grounding).
    split: HF split for ``mode='chartqa'`` (defaults train/val by ``training``).
    num_examples: Cap on dataset size (subset for smoke runs).
    color_seed: Seed for the synthetic color task.
    query_max_len: Max question tokens kept.
    num_workers: grain worker processes (0 = synchronous; transforms are
      pure-numpy so any value is safe).

  Returns:
    A ``ChartQADataset`` config ready to plug into ``cfg.train_ds`` /
    ``cfg.eval_ds``.
  """
  if batch_size != 1:
    raise ValueError(
        "Multimodal ChartQA SFT requires batch_size=1 per process (Gemma4 "
        "vision merge is single-sequence); got batch_size="
        f"{batch_size}."
    )
  tokenizer = gm.text.Gemma4Tokenizer()
  st = gm.text.Gemma4Tokenizer.special_tokens
  if split is None:
    split = "train" if training else "val"

  transforms = [
      BuildChartQAInputs(
          tokenizer=tokenizer,
          prompt_len=prompt_len,
          n_soft=n_soft,
          image_size=image_size,
          patch_size=patch_size,
          query_max_len=query_max_len,
          bos_token=int(st.BOS),
          soi_token=int(st.START_OF_IMAGE),
          eoi_token=int(st.END_OF_IMAGE),
          pad_token=int(st.PAD),
          pooling_kernel_size=pooling_kernel_size,
          max_soft_tokens=max_soft_tokens,
          # (eval only) ground-truth answer tokens for the ChartQA metric.
          answer_len=0 if training else answer_len,
      ),
      adapter_data.CanvasChunker(
          in_response="response",
          out_canvas="canvas",
          out_canvas_id="canvas_id",
          out_canvas_mask="canvas_mask",
          num_canvases=num_canvases,
          canvas_size=canvas_size,
          eos_token=int(st.EOS),
          pad_token=int(st.PAD),
      ),
      ChartQAEncoderTargets(
          pad_token=int(st.PAD),
          invalid_ids=(
              TOKEN_PLACEHOLDER,
              int(st.START_OF_IMAGE),
              int(st.END_OF_IMAGE),
          ),
      ),
      kd.data.Rearrange(key="canvas", pattern="c -> c 1"),
  ]

  keep_fields = [
      "patches",
      "positions_xy",
      "prompt",
      "canvas",
      "canvas_id",
      "canvas_mask",
      "encoder_target",
      "encoder_target_mask",
  ]
  if not training:
    keep_fields.append("answer_tokens")  # ground truth for the ChartQA metric
  transforms.append(kd.data.Elements(keep=keep_fields))

  return ChartQADataset(
      mode=mode,
      split=split,
      num_examples=num_examples,
      image_size=image_size,
      color_seed=color_seed,
      shuffle=training,
      num_epochs=None if training else 1,
      batch_size=batch_size,
      num_workers=num_workers,
      transforms=transforms,
  )
