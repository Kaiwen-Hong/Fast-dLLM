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

"""Custom network, model and losses for DiffusionGemma SFT."""

import dataclasses
from typing import Any

import flax
import flax.linen as nn
from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_ar_state_handler
from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_network
from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput
from gemma.gm.vision import _token_utils
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import inference
from hackable_diffusion.lib import sampling
import jax
import jax.numpy as jnp
from kauldron import kd
import optax

from gemma.diffusion.hackable_diffusion_adapter import checkpointed_evaluator as _checkpointed_evaluator  # pylint: disable=line-too-long
CheckpointedEvaluator = _checkpointed_evaluator.CheckpointedEvaluator


PAD_TOKEN = 0


def _build_vision_input(patches, positions_xy, n_soft):
  """Pack batched per-example patches into a single ``PreprocessedVisionInput``.

  grain yields ``patches`` ``[B, MAX_PATCHES, P]`` and ``positions_xy``
  ``[B, MAX_PATCHES, 2]``. The Gemma4 vision merge (``_encode_vision``) expects a
  single meta-batch ``[1, n_images*MAX_PATCHES, P]`` with ``soft_token_counts``
  enumerating the images. The vendored ``merge_flat_embeddings`` vmaps the text
  batch against a size-1 vision batch, so it only reconciles when the per-call
  text batch is 1 — hence the multimodal SFT configs use ``batch_size=1``. The
  reshape below is the identity for ``B==1`` and matches the canonical packing in
  ``gm/text/_gemma4_sampler.py``. Returns ``None`` for text-only (no patches).
  """
  if patches is None:
    return None
  n_images = patches.shape[0]
  max_patches = patches.shape[1]
  flat_patches = jnp.reshape(
      patches, (1, n_images * max_patches, patches.shape[2])
  )
  flat_positions = jnp.reshape(
      positions_xy, (1, n_images * max_patches, positions_xy.shape[2])
  )
  return PreprocessedVisionInput(
      patches=flat_patches,
      positions_xy=flat_positions,
      soft_token_counts=(n_soft,) * n_images,
  )


def shift_encoder_targets_for_multimodal(
    *,
    encoder_target: jnp.ndarray,
    encoder_target_mask: jnp.ndarray,
    prompt: jnp.ndarray,
    x0_tokens: jnp.ndarray,
    l_no_mm: int,
    num_tokens_per_image: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Shifts encoder targets to align with non-multimodal token indices.

  Internal-parity implementation (verbatim from the internal SFT stack — see
  discreteGemma/response-from-jestki.md Doc 2 §6b). The encoder logits come
  back MM-stripped ([B, L_no_mm, V] via ``remove_mm_logits``); this gathers the
  full-length targets onto the SAME no-MM indexing (both use
  ``_token_utils.get_no_mm_indices``), so the AR CE is computed on text
  positions only.

  Args:
    encoder_target: Original targets, shape [B, L].
    encoder_target_mask: Original target mask, shape [B, L].
    prompt: Prompt tokens (expanded, containing the image block), [B, PromptLen].
    x0_tokens: Clean canvas tokens, shape [B, TotalCanvasLen].
    l_no_mm: Expected output length (after removing MM tokens).
    num_tokens_per_image: Max tokens per image.

  Returns:
    A tuple of (shifted_encoder_target, shifted_encoder_target_mask).
  """
  full_seq = jnp.concatenate([prompt, x0_tokens], axis=1)
  new_positions = _token_utils.get_no_mm_indices(
      tokens=full_seq,
      l_no_mm=l_no_mm,
      num_tokens_per_image=num_tokens_per_image,
  )

  shifted_encoder_target = jnp.take_along_axis(
      encoder_target, new_positions, axis=1
  )
  shifted_encoder_target_mask = jnp.take_along_axis(
      encoder_target_mask, new_positions, axis=1
  )
  return shifted_encoder_target, shifted_encoder_target_mask


@dataclasses.dataclass(kw_only=True, frozen=True)
class SFTInferenceFn(inference.InferenceFn):
  """Inference function that applies the Gemma network for denoising."""

  gemma_network: Any
  params: Any

  def __call__(
      self,
      time: hd_typing.TimeTree,
      xt: hd_typing.DataTree,
      conditioning: hd_typing.Conditioning | None,
  ) -> hd_typing.TargetInfoTree:
    """Evaluates the denoiser backbone at the given diffusion time and state.

    Args:
      time: Diffusion time steps.
      xt: Noisy canvas tokens or array.
      conditioning: Auxiliary conditioning dict containing cache, positions, and
        masks.

    Returns:
      A tree containing the predicted denoising targets (logits).
    """
    if conditioning and 'sc_logits' in conditioning:
      sc_logits = conditioning['sc_logits']
      if sc_logits.ndim == 4 and sc_logits.shape[-2] == 1:
        conditioning = {
            **conditioning,
            'sc_logits': jnp.squeeze(sc_logits, axis=-2),
        }

    output = self.gemma_network.apply(
        {'params': self.params},
        xt=xt,
        time=time,
        conditioning=conditioning,
        is_training=False,
    )
    logits = output['logits']
    return {
        'logits': logits,
    }


################################################################################
# MARK: Decoder helpers
################################################################################


def sft_decode(
    gemma_network: Any,
    *,
    xt: hd_typing.DataTree,
    time: hd_typing.TimeTree,
    kv_cache: Any,
    positions: jnp.ndarray,
    prompt_mask: jnp.ndarray,
    canvas_mask: jnp.ndarray,
    selected_canvas_idx: jnp.ndarray,
    prompt_len: int,
    total_canvas_len: int,
    canvas_size: int,
    sc_logits: jnp.ndarray | None = None,
    is_training: bool = True,
) -> hd_typing.TargetInfoTree:
  """Runs the SFT decoder pass: denoise canvases using the prefilled KV cache.

  Builds an attention mask where each canvas token attends to the prompt plus
  all canvases up to and including the selected canvas.

  Args:
    gemma_network: The Gemma backbone wrapper (bound or unbound).
    xt: Noised canvas tokens.
    time: Diffusion time.
    kv_cache: Prefilled KV cache.
    positions: Full-sequence positions.
    prompt_mask: Non-pad prompt mask.
    canvas_mask: Valid-canvas mask, shape ``[B, TotalCanvasLen]``.
    selected_canvas_idx: Per-example selected canvas index, shape ``[B]``.
    prompt_len: Fixed padded maximum prompt length.
    total_canvas_len: Total canvas length.
    canvas_size: Number of tokens per canvas.
    sc_logits: Optional self-conditioning logits.
    is_training: Whether we are in training mode.

  Returns:
    Denoiser output dict (e.g., ``{'logits': logits}``).
  """
  attn_mask = mask_helpers.create_decoder_attention_mask(
      prompt_mask=prompt_mask,
      canvas_mask=canvas_mask,
      selected_canvas_idx=selected_canvas_idx,
      prompt_len=prompt_len,
      total_canvas_len=total_canvas_len,
      canvas_size=canvas_size,
      num_queries=total_canvas_len,
  )

  # Build canvas positions.
  canvas_positions = positions[:, prompt_len:]

  # Build conditioning dict.
  conditioning = {
      'kv_cache': kv_cache,
      'positions': canvas_positions,
      'attention_mask': attn_mask,
  }
  if sc_logits is not None:
    conditioning['sc_logits'] = sc_logits

  return gemma_network(
      xt=xt,
      time=time,
      conditioning=conditioning,
      is_training=is_training,
  )


def sft_encode(
    gemma_network: Any,
    *,
    prompt: jnp.ndarray,
    x0_tokens: jnp.ndarray,
    canvas_mask: jnp.ndarray,
    selected_canvas_idx: jnp.ndarray,
    prompt_len: int,
    total_canvas_len: int,
    canvas_size: int,
    pad_token: int = PAD_TOKEN,
    images: Any = None,
) -> tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray]:
  """Runs the SFT encoder pass: prefill KV cache and compute encoder logits.

  Note: ``total_canvas_len`` is unused here but accepted to maintain a
  consistent
  function signature with ``sft_decode``.

  Args:
    gemma_network: The Gemma backbone wrapper (bound or unbound).
    prompt: Prompt tokens.
    x0_tokens: Target response tokens.
    canvas_mask: Valid-canvas mask, shape ``[B, TotalCanvasLen]``.
    selected_canvas_idx: Per-example selected canvas index, shape ``[B]``.
    prompt_len: Fixed padded maximum prompt length.
    total_canvas_len: Total canvas length.
    canvas_size: Number of tokens per canvas.
    pad_token: Padding token ID.

  Returns:
    A tuple containing:
      - encoder_logits: Logits of the sequence tokens under the encoder.
      - kv_cache: Prefilled KV cache with set end index.
      - positions: Positional offset IDs.
      - prompt_mask: Mask filtering out pad tokens in prompt.
  """
  del total_canvas_len  # Unused; accepted for API consistency.
  # Concatenate prompt and clean canvas tokens.
  full_seq = jnp.concatenate([prompt, x0_tokens], axis=1)  # [B, FullSeqLen]
  # Mask out PAD tokens.
  prompt_mask = prompt != pad_token  # [B, PromptLen]
  full_seq_mask = jnp.concatenate([prompt_mask, canvas_mask], axis=1)

  kv_cache, encoder_logits, positions, _ = (
      hd_gemma_network.prefill_kv_cache_with_encoder(
          tokens=full_seq,
          input_mask=full_seq_mask,
          init_cache_fn=gemma_network.init_cache,
          encoder_fn=gemma_network.encoder_call,
          # IMAGE: merge vision soft tokens at the -2 placeholders in `prompt`
          # during the (prompt+clean-canvas) prefill. None => text-only.
          images=images,
      )
  )

  # Set end_index per example: prompt_len + selected_canvas_idx * canvas_size.
  # This makes the decoder reuse cached K/V for the prompt and all canvases
  # before the selected one.
  end_index = prompt_len + selected_canvas_idx * canvas_size  # [B]
  kv_cache = mask_helpers.set_cache_end_index(kv_cache, end_index)

  return encoder_logits, kv_cache, positions, prompt_mask


################################################################################
# MARK: SFTDiffusion
################################################################################


class SFTDiffusion(nn.Module):
  """Custom Kauldron model for Supervised Fine-Tuning (SFT) training.

  This model wraps the Gemma model in a discrete-time categorical diffusion
  process for token generation. It handles time sampling, noise corruption,
  self-conditioning, and computes both denoiser SFT predictions and encoder AR
  logits.

  Attributes:
    gemma_network: The underlying wrapped Gemma diffusion network model.
    corruption_process: The noise process used to corrupt inputs.
    time_sampler: Time sampler for generating training noise.
    prompt_len: Max sequence length for prompt tokens.
    canvas_size: Token length of each canvas/generation segment.
    num_canvases: The number of canvas segments to generate/denoise.
    x0: Context key pointing to the target canvas.
    prompt: Context key pointing to the input prompt tokens.
    canvas_id: Context key pointing to the segment identifiers.
    canvas_mask: Context key pointing to active tokens in the canvas.
    encoder_target: Context key pointing to the encoder target tokens.
    encoder_target_mask: Context key pointing to the active target token masks.
    pad_token: Token value used for padding.
    stop_gradient_from_denoiser_to_encoder: Whether to block gradients between
      the denoiser head and the encoder base.
    self_cond_prob: Probability of using self-conditioning during training.
  """

  gemma_network: Any
  corruption_process: Any
  time_sampler: Any

  prompt_len: int
  canvas_size: int
  num_canvases: int

  # Kontext keys
  x0: kd.kontext.Key
  prompt: kd.kontext.Key
  canvas_id: kd.kontext.Key
  canvas_mask: kd.kontext.Key
  encoder_target: kd.kontext.Key
  encoder_target_mask: kd.kontext.Key

  pad_token: int = PAD_TOKEN
  stop_gradient_from_denoiser_to_encoder: bool = False
  self_cond_prob: float = 0.5
  # Optional multimodal keys. None => text-only (sudoku/pubmedqa unchanged).
  # `patches`/`positions_xy` come from the data pipeline; `n_soft` is the number
  # of vision soft tokens per image (== count of -2 placeholders in the prompt).
  patches: kd.kontext.Key | None = None
  positions_xy: kd.kontext.Key | None = None
  n_soft: int = 0

  @property
  def total_canvas_len(self) -> int:
    return self.num_canvases * self.canvas_size

  @nn.compact
  def __call__(
      self,
      x0: jnp.ndarray,
      prompt: jnp.ndarray,
      canvas_id: jnp.ndarray,
      canvas_mask: jnp.ndarray,
      encoder_target: jnp.ndarray,
      encoder_target_mask: jnp.ndarray,
      patches: jnp.ndarray | None = None,
      positions_xy: jnp.ndarray | None = None,
      is_training: bool = True,
  ):
    """Computes model losses and forward predictions during training or eval.

    Args:
      x0: Int array [B, SeqLen] of initial target canvas states.
      prompt: Int array [B, PromptLen] of input prompt sequences.
      canvas_id: Int array [B, SeqLen] of segment identifier tokens.
      canvas_mask: Float/Int array [B, SeqLen] masking active (non-pad) canvas
        positions.
      encoder_target: Int array [B, SeqLen] of target token IDs.
      encoder_target_mask: Float/Int array [B, SeqLen] of target masks.
      is_training: If True, operates in training mode (enabling dropout, etc.).

    Returns:
      A dictionary containing logits, targets, predictions, and loss metadata.
    """
    ############################################################################
    # Sample time and corrupt x0
    ############################################################################

    # Sample time
    time = self.time_sampler(self.make_rng('sampling'), x0)

    # Corrupt x0
    xt, target_info = self.corruption_process.corrupt(
        self.make_rng('sampling'), x0, time
    )

    ############################################################################
    # Sample canvas
    ############################################################################

    # Sample which canvas to train on
    # Count valid canvases per example by checking the first token of each
    # canvas in canvas_mask.
    first_token_indices = jnp.arange(self.num_canvases) * self.canvas_size
    canvas_validity = canvas_mask[:, first_token_indices]  # [B, num_canvases]
    num_valid_canvases = jnp.sum(canvas_validity, axis=-1)  # [B]
    # Clip to at least 1 to avoid zero-division on empty examples.
    num_valid_canvases = jnp.maximum(num_valid_canvases, 1)

    # Uniformly sample a canvas index from [0, num_valid_canvases) per example.
    rng_canvas = self.make_rng('sampling')
    selected_canvas_idx = jax.random.randint(
        rng_canvas,
        shape=num_valid_canvases.shape,
        minval=0,
        maxval=num_valid_canvases,
    )  # [B]

    # Squeeze trailing dim if present (hackable diffusion uses <B, L, 1>).
    x0_tokens = x0[..., 0] if x0.ndim == 3 else x0  # [B, TotalCanvasLen]

    ############################################################################
    # Create KV cache and encoder logits
    ############################################################################

    # IMAGE: build the vision input from batched patches (None => text-only) and
    # merge it into the encoder prefill at the -2 placeholders in `prompt`.
    images = _build_vision_input(patches, positions_xy, self.n_soft)

    encoder_logits, kv_cache, positions, prompt_mask = sft_encode(
        gemma_network=self.gemma_network,
        prompt=prompt,
        x0_tokens=x0_tokens,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        prompt_len=self.prompt_len,
        total_canvas_len=self.total_canvas_len,
        canvas_size=self.canvas_size,
        pad_token=self.pad_token,
        images=images,
    )

    if self.stop_gradient_from_denoiser_to_encoder:
      kv_cache = jax.lax.stop_gradient(kv_cache)

    ############################################################################
    # Decode first pass
    ############################################################################

    decoder_kwargs = dict(
        gemma_network=self.gemma_network,
        xt=xt,
        time=time,
        kv_cache=kv_cache,
        positions=positions,
        prompt_mask=prompt_mask,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        prompt_len=self.prompt_len,
        total_canvas_len=self.total_canvas_len,
        canvas_size=self.canvas_size,
        is_training=is_training,
    )

    denoiser_output_first_pass = sft_decode(**decoder_kwargs)

    # Derive target_mask: only the selected canvas contributes to loss
    target_mask = canvas_mask & (canvas_id == selected_canvas_idx[:, None])

    # Combine is_corrupted with target_mask to ignore non-selected tokens
    target_info['is_corrupted'] = (
        target_info['is_corrupted'] & target_mask[..., None]
    )
    target_info['target_mask'] = target_mask[..., None]

    # Convert predictions (computes loss-ready dict)
    converted_first_pass = self.corruption_process.convert_predictions(
        denoiser_output_first_pass, xt, time
    )

    ############################################################################
    # Self-conditioning & decode second pass
    ############################################################################

    converted_first_pass = jax.lax.stop_gradient(converted_first_pass)
    sc_logits = converted_first_pass['logits']
    zero_logits = jnp.zeros_like(sc_logits)

    # With probability self_cond_prob, run self-conditioning element-wise.
    batch_size = xt.shape[0]
    do_self_cond = (
        jax.random.uniform(self.make_rng('sampling'), shape=(batch_size,))
        < self.self_cond_prob
    )
    # Reshape to broadcast with x0_hat_logits (Batch, ..., Channels)
    do_self_cond = do_self_cond.reshape(
        (batch_size,) + (1,) * (sc_logits.ndim - 1)
    )
    sc_logits = jnp.where(do_self_cond, sc_logits, zero_logits)

    denoiser_output = sft_decode(**decoder_kwargs, sc_logits=sc_logits)

    # Convert predictions (computes loss-ready dict)
    converted = self.corruption_process.convert_predictions(
        denoiser_output, xt, time
    )

    # Get noise info
    noise_info = self.corruption_process.get_schedule_info(time)

    # IMAGE (internal parity): with images, encoder_logits come back MM-stripped
    # to [B, L_no_mm, V] (remove_mm_logits inside the gemma forward); gather the
    # full-length targets onto the same no-MM indexing before the AR CE.
    if images is not None:
      encoder_target, encoder_target_mask = (
          shift_encoder_targets_for_multimodal(
              encoder_target=encoder_target,
              encoder_target_mask=encoder_target_mask,
              prompt=prompt,
              x0_tokens=x0_tokens,
              l_no_mm=encoder_logits.shape[1],
              num_tokens_per_image=self.n_soft,
          )
      )

    return {
        'output': converted,
        'target': target_info,
        'xt': xt,
        'noise_info': noise_info,
        'encoder_logits': encoder_logits,
        'encoder_target': encoder_target,
        'encoder_target_mask': encoder_target_mask,
    }


################################################################################
# MARK: SamplingEvaluator
################################################################################


class GemmaKDARSampler(sampling.AutoregressiveDiffusionSampler):
  """Custom sampler for DiffusionGemma SFT."""

  state_handler: hd_gemma_ar_state_handler.GemmaARStateHandler

  def update_from_context(self, context: kd.train.Context):
    """Updates the sampler from a Kauldron context."""
    self.state_handler.update_from_context(context)


class GemmaSamplingEvaluator(CheckpointedEvaluator):
  """Custom SamplingEvaluator for DiffusionGemma SFT."""

  canvas_sampler: sampling.SampleFn | None = None
  ar_diffusion_sampler: GemmaKDARSampler | None = None
  num_batches: int | None

  gemma_network_path: str = 'gemma_network'

  pad_token: int = PAD_TOKEN

  rng_stream: str = 'sampling'

  # override the default values for losses, metrics, and summaries to be empty
  # (because the training ones likely don't make sense for sampling)
  losses: dict[str, kd.losses.Loss] = dataclasses.field(default_factory=dict)
  metrics: dict[str, kd.metrics.Metric] = dataclasses.field(
      default_factory=dict
  )
  summaries: dict[str, kd.metrics.Metric] = dataclasses.field(
      default_factory=dict
  )

  # Set the default checkpointer to a noop checkpointer.
  checkpointer: kd.ckpts.BaseCheckpointer = kd.ckpts.NoopCheckpointer()

  def _make_inference_fn(
      self, model: nn.Module, context: kd.train.Context
  ) -> inference.InferenceFn:
    """Build an ``SFTInferenceFn`` from the model and context.

    Args:
      model: The SFT diffusion model module.
      context: The Kauldron training/eval context.

    Returns:
      The bound inference function for diffusion sampling.
    """
    gemma_network = kd.kontext.get_by_path(model, self.gemma_network_path)
    params = kd.kontext.get_by_path(context.params, self.gemma_network_path)
    return SFTInferenceFn(gemma_network=gemma_network, params=params)

  def _step(
      self, step_nr: int, state: kd.train.TrainState, batch: Any
  ) -> kd.train.AuxiliariesState:
    """Custom sampling eval step.

    Args:
      step_nr: Current evaluation step number.
      state: Kauldron training state containing model parameters.
      batch: The evaluation data batch.

    Returns:
      Auxiliaries state containing generated samples and latency summaries.
    """
    return self._ar_diffusion_step(step_nr, state, batch)

  def _ar_diffusion_step(
      self, step_nr: int, state: kd.train.TrainState, batch: Any
  ) -> kd.train.AuxiliariesState:
    """Runs full AR diffusion token generation using the JAX/Kauldron sampler.

    Args:
      step_nr: Current evaluation step number.
      state: Kauldron training state containing model parameters.
      batch: The evaluation data batch.

    Returns:
      Auxiliaries state containing generated samples and latency summaries.
    """
    # Set up the context and the inference function
    base_context = kd.train.Context.from_state_and_batch(
        state=state, batch=batch
    )
    context = SamplingContext(**base_context.__dict__)
    inference_fn = self._make_inference_fn(self.model, context)
    # Update the context of the AR sampler.
    assert self.ar_diffusion_sampler is not None
    self.ar_diffusion_sampler.update_from_context(context)

    # Create PRNG keys for init and sampling
    rngs = self.base_cfg.rng_streams.eval_rngs(step_nr)
    _, sample_rng = jax.random.split(rngs[self.rng_stream], 2)

    _, kwargs = kd.data.utils.get_model_inputs(self.model, context)
    prompt_tokens = kwargs['prompt']
    # In case when there is no padding, this will work correctly when B=1.
    prompt_lengths = jnp.sum(prompt_tokens != self.pad_token, axis=-1)  # [B]

    cond = {
        'prompt_tokens': prompt_tokens,
        'prompt_lengths': prompt_lengths,
    }
    # IMAGE: if the model is multimodal, image-condition the AR sampler's encoder
    # prefill by packing patches -> PreprocessedVisionInput into the conditioning
    # (consumed by GemmaARStateHandler.init_ar_state). Text-only tasks omit it.
    patches = kwargs.get('patches', None)
    if patches is not None:
      cond['images'] = _build_vision_input(
          patches, kwargs.get('positions_xy'), getattr(self.model, 'n_soft', 0)
      )

    # Run the sampling loop
    final, final_state = self.ar_diffusion_sampler(
        diffusion_inference_fn=inference_fn,
        batch_size=len(prompt_tokens),
        rng=sample_rng,
        conditioning=cond,
    )
    final = jnp.expand_dims(final, axis=-1)

    processed_denoising_steps = jnp.array(
        final_state['processed_denoising_steps'], dtype=jnp.float32
    ).reshape(())
    processed_num_canvases = jnp.array(
        final_state['processed_num_canvases'], dtype=jnp.float32
    ).reshape(())
    average_denoising_steps_per_canvas = (
        processed_denoising_steps / processed_num_canvases
    )
    average_denoising_steps_per_canvas = jnp.array(
        average_denoising_steps_per_canvas, dtype=jnp.float32
    ).reshape(())

    # Update the context with the final and intermediate samples
    # final and interms are DiffusionStep trees.
    context = context.replace(
        samples=final,
        # Latency metrics.
        processed_denoising_steps=processed_denoising_steps,
        processed_num_canvases=processed_num_canvases,
        average_denoising_steps_per_canvas=average_denoising_steps_per_canvas,
    )
    # Compute the metrics
    context = self.aux.update_context(context)
    return context.get_aux_state(
        return_losses=True, return_metrics=True, return_summaries=True
    )

  def __hash__(self) -> int:
    # Make Evaluator hashable, so its methods can be jitted.
    return id(self)


################################################################################
# MARK: SamplingContext
################################################################################


@flax.struct.dataclass
class SamplingContext(kd.train.Context):
  """Context with additional fields for sampling."""

  samples: Any = None
  processed_denoising_steps: Any = None
  processed_num_canvases: Any = None
  average_denoising_steps_per_canvas: Any = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class EncoderARLoss(kd.losses.Loss):
  """Causal loss for the encoder."""

  encoder_logits: kd.kontext.Key = 'preds.encoder_logits'
  encoder_target: kd.kontext.Key = 'preds.encoder_target'
  encoder_target_mask: kd.kontext.Key = 'preds.encoder_target_mask'

  def get_values(
      self,
      encoder_logits: jnp.ndarray,
      encoder_target: jnp.ndarray,
      encoder_target_mask: jnp.ndarray,
  ) -> jnp.ndarray:
    """Computes the masked per-example cross-entropy loss for the encoder.

    Args:
      encoder_logits: Predicted encoder logits of shape [B, S, V].
      encoder_target: Target token IDs of shape [B, S].
      encoder_target_mask: Loss weight mask of shape [B, S].

    Returns:
      Per-example average cross-entropy loss of shape [B].
    """
    # encoder_logits shape: [B, FullSeqLen, VocabSize]
    # encoder_target shape: [B, FullSeqLen]
    # encoder_target_mask shape: [B, FullSeqLen]

    # Compute cross entropy
    loss = optax.softmax_cross_entropy_with_integer_labels(
        encoder_logits, encoder_target
    )  # [B, FullSeqLen]

    # Apply mask
    masked_loss = loss * encoder_target_mask

    # Average over valid positions per example
    sum_loss = jnp.sum(masked_loss, axis=-1)
    sum_mask = jnp.sum(encoder_target_mask, axis=-1)

    # Avoid division by zero
    per_example_loss = sum_loss / jnp.maximum(sum_mask, 1.0)

    return per_example_loss
