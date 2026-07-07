# Responses from the internal coding agent ("jestki")

> Archived verbatim by Claude Code, 2026-07-07. Two documents received from the internal coding
> agent, reproduced unmodified below. These are the **[internal·doc]** / **[internal·reply]**
> provenance sources cited by `0706-design-doc-prompts.md`.
>
> - **Doc 1** — "DiffusionGemma Base vs. Internal SFT Codebase Comparison" (received first;
>   describes the internal SFT/multimodal codebase vs the external release).
> - **Doc 2** — "DiffusionGemma SFT Data-Pipeline & Evaluation Reference" (answers Q-D3 / Q-D5
>   from `0706-design-doc-prompts.md` §10).
>
> ⚠️ Known discrepancy to resolve at Phase A (see design doc §8): Doc 2 reports Gemma4 special
> ids `IMAGE_PLACEHOLDER=258880`, `END_OF_IMAGE=258882`, while a block in the local pristine
> `gm/text/_tokenizer.py` shows `IMAGE_PLACEHOLDER=255999`, `END_OF_IMAGE=256000` — assert the
> local `Gemma4Tokenizer.special_tokens` against Doc 2's table before trusting either.

---
---

# Doc 1 — DiffusionGemma Base vs. Internal SFT Codebase Comparison

This document provides a detailed, technical comparison of the **original external DiffusionGemma release** and the **internal SFT/multimodal codebase** in this workspace.

---

## 1. Vision Encoder Integration (Multimodality)

The external DiffusionGemma weights support multimodal inputs. However, the external `gemma` repository is primarily text-only, and open-weights fine-tuning (e.g. in NeMo or Hugging Face) uses standard external wrappers.

In this internal codebase (located at `third_party/py/gemma/diffusion/hackable_diffusion_adapter`), multimodality is integrated directly into the JAX/Flax transformer execution graph through an adapter model and extensive monkey-patching of the underlying Gemma library modules.

### A. Model Wrapper and Configuration

- **`PrimeDiffusionGemmaModel`** (`hd_gemma_network.py`):
  - Directly extends `gemma_diffusion.DiffusionGemma_26B_A4B` (the core text diffusion model).
  - Automatically instantiates and binds the vision encoder if it is missing from the configuration:

```python
vision_encoder = gemma_vision.VisionEncoder(
    name="vision_encoder",
    d_model=1152,
    num_layers=27,
    num_heads=16,
    ffw_hidden=4304,
    output_length=280,  # Max patches/soft tokens per image
    use_clipped_linears=False,
    standardize_embeddings=True,
)
```

### B. Transformer & JAX Patching Details

To weave the vision encoder inputs into the text generation pipeline without altering the core Gemma library files, several key monkey-patches are applied in `hd_gemma_network.py`:

1. **`Transformer.__getattr__` Override:**
   - Allows the model to resolve the sub-module `"vision_encoder"` on-the-fly, returning it from either child modules or the configuration:

```python
def custom_getattr(self, name: str):
    if name == "vision_encoder":
        if "vision_encoder" in self._children:
            return self._children["vision_encoder"]
        return self.config.vision_encoder
    return _orig_getattr(self, name)
```

2. **Multimodal Embedding Merging (`_merge_mm_embeddings`):**
   - Replaces the default text embedding merge function on the Transformer class with `custom_merge_mm_embeddings`.
   - It calls `self._encode_vision(images)` to compute the image embeddings (`soft_embeddings`).
   - If the batch size exceeds 1 while `soft_embeddings` has a batch dimension of 1, it automatically reshapes and broadcasts `soft_embeddings` to match the batch size.
   - It locates the `<|image|>` placeholder tokens in the input sequence using `tokens == TOKEN_PLACEHOLDER`, and inserts the visual embeddings using `_token_utils.merge_flat_embeddings`.

3. **Positional Embeddings (`factorized_posemb`):**
   - Patched with `safe_factorized_posemb` on `gemma4_images`.
   - If the positional embedding tensor is 4D (i.e. `[H, W, 2, dim]`), it reshapes it to 3D `[-1, 2, dim]`.
   - It maps the 2D pixel coordinates (`positions_xy`) to the embedding coordinates via JAX matrix multiplication, ensuring spatial relations are properly aligned when computing attention over image tokens.

4. **JAX Tree Map Shielding (`tree_map_with_path`):**
   - Patched with `custom_tree_map_with_path`.
   - JAX's standard PyTree utilities attempt to traverse model arguments during tracing. This patch intercepts and shields complex non-numeric fields (such as `self`, `images`, and `audio` keys) from being traversed, avoiding JAX compiler/tracing crashes.

5. **Attention Mask Alignment (`_encode_and_get_inputs`):**
   - Patched with `custom_encode_and_get_inputs` on `gemma4_transformer.Transformer`.
   - In cases where the padded `sliding_attention_mask` and full `attention_mask` have mismatching token lengths, it overrides the sliding mask with the caller's attention mask to prevent JAX shape-assertion failures.

6. **Memory Optimizations (`nn.remat` / Gradient Checkpointing):**
   - Patches block calls on both `gemma4_modules.Block` and `gemma_vision_modules.Block` to execute with Flax rematerialization (`nn.remat`).
   - This discards intermediate activations during the forward pass and recomputes them during backpropagation, dramatically saving TPU device memory (HBM) and preventing Out-Of-Memory (OOM) errors.

---

## 2. Trainer Framework (Kauldron vs. External)

The training codebase uses **Kauldron** (Google's JAX/Flax-based training framework) to configure and run the optimization loop:

- **Trainer Config (`sft_trajectory.py`):**
  - Instantiates `kd.train.Trainer()`.
  - Configures FSDP (Fully Sharded Data Parallel) sharding strategies for both model parameters and optimizer state:

```python
cfg.sharding = kd.sharding.ShardingStrategy(
    params=kd.sharding.FSDPSharding(),
    opt_state=kd.sharding.FSDPSharding(),
)
```

- **Custom Dual-Task Loss:**
  - **Diffusion Loss:** A categorical discrete diffusion loss (`NoWeightDiscreteLoss`) computed over the masked targets on the generation canvas.
  - **Encoder Loss (`EncoderARLoss`):** A standard causal cross-entropy loss applied over the prompt and context tokens to train the causal prefill encoder.

- **TPU Logging Optimization (`TensorBoardOnlyWriter`):**
  - Configured inside the training configuration (e.g. `my_sft_trajectory.py`).
  - By checking for the Borg environment variable `XM_XID`, it dynamically overrides the Kauldron metric writer with `TensorBoardOnlyWriter` to suppress heavy Datatable writes, only exporting lightweight summaries to TensorBoard and stdout logs.

---

## 3. Data Processing Pipeline & Target Alignment

The data pipeline (in `trajectory_data.py` and `chartqa_data.py`) uses **Grain** (PyGrain) data loaders instead of Hugging Face dataset structures:

### A. Dataset Loading and Formats

- Custom Grain `RandomAccessDataSource` layers (such as `TrajectoryBagzDataSource` or `ChartQAArrayRecordDataSource`) read records from CNS files.
- Records are parsed from raw bytes containing `raw_image`, `question`, and `answer`.

### B. Preprocessing Transforms

1. **`PreprocessChartQAVision`:**
   - Executes image scaling and patchification entirely on CPU numpy using PIL:

```python
patches, positions_xy, soft_token_counts = (
    _preprocessing.preprocess_and_patchify(
        [pil_img],
        patch_size=self.patch_size,
        max_soft_tokens=self.max_soft_tokens,
        pooling_kernel_size=self.pooling_kernel_size,
    )
)
```

   - Running this on CPU beforehand bypasses JAX multiprocessing locks on worker nodes.

2. **`ExpandImagePlaceholder`:**
   - Expands the single `<|image|>` prompt token into a sequence of placeholder tokens matching the size of the vision encoder's output (`280` soft tokens + `3` boundary/newline tokens).

3. **`CanvasChunker`:**
   - Splits the target text response into a sequence of `num_canvases` of size `canvas_size` (producing `canvas`, `canvas_id`, and `canvas_mask`), enabling discrete block diffusion.

### C. Multimodal Target Alignment (`shift_encoder_targets_for_multimodal`)

Because image placeholder expansion inserts 283 tokens into the input sequence, the computed logits and the targets would become misaligned:

- The autoregressive cross-entropy loss is only computed on text tokens.
- `encoder_logits` outputs have a shape of `[B, L_no_mm, V]` (where `L_no_mm` represents the sequence length excluding multimodal image tokens).
- `encoder_target` and `encoder_target_mask` (created from `SequenceTargetShift`) have the full expanded length `[B, L_expanded]`.
- To compute the loss, `shift_encoder_targets_for_multimodal` aligns the targets by extracting only the indices of non-multimodal tokens via `_token_utils.get_no_mm_indices`:

```python
new_positions = _token_utils.get_no_mm_indices(
    tokens=full_seq,
    l_no_mm=l_no_mm,
    num_tokens_per_image=num_tokens_per_image,
)
shifted_encoder_target = jnp.take_along_axis(encoder_target, new_positions, axis=1)
```

---
---

# Doc 2 — DiffusionGemma SFT Data-Pipeline & Evaluation Reference

This document answers the technical specifications for the internal DiffusionGemma SFT data pipeline (`Q-D3`) and evaluation/sampling configuration (`Q-D5`) with exact code snippets and constants extracted from the workspace.

---

## Part 1: Data-Pipeline Schema (Q-D3)

### 1. Record Format

#### a. On-disk serialization

Yes, one on-disk record is a serialized `tf.train.Example`.

#### b. Exact feature keys, types, and encodings

On disk, each `tf.train.Example` contains the following features:

- `image/encoded`: `bytes_list` holding the raw image bytes encoded in JPEG or PNG format.
- `question`: `bytes_list` holding the question text encoded in `utf-8` bytes.
- `answer`: `bytes_list` holding the answer text encoded in `utf-8` bytes.

#### c. Parse functions

The parse functions used by the Grain pipelines to deserialize the records are as follows:

For **ChartQA** (`ParseChartQARecord` in `chartqa_data.py`):

```python
class ParseChartQARecord(grain.MapTransform):
    """Parses tf.train.Example from ArrayRecord bytes."""

    def map(self, raw_bytes: bytes) -> dict[str, Any]:
        example = tf.train.Example.FromString(raw_bytes)
        feature = example.features.feature

        image_bytes = feature["image/encoded"].bytes_list.value[0]
        question = feature["question"].bytes_list.value[0].decode("utf-8")
        answer = feature["answer"].bytes_list.value[0].decode("utf-8")

        return {
            "raw_image": image_bytes,
            "prompt": question,
            "prompt_text": question,
            "short_answer": answer,
            "short_answer_text": answer,
            "response_text": f"The answer is: {answer}",
        }
```

For **Trajectory** (`ParseTrajectoryRecord` in `trajectory_data.py`):

```python
class ParseTrajectoryRecord(grain.MapTransform):
    """Parses tf.train.Example from Bagz bytes."""

    def map(self, raw_bytes: bytes) -> dict[str, Any]:
        example = tf.train.Example.FromString(raw_bytes)
        feature = example.features.feature

        image_bytes = feature["image/encoded"].bytes_list.value[0]
        question = feature["question"].bytes_list.value[0].decode("utf-8")
        answer = feature["answer"].bytes_list.value[0].decode("utf-8")

        return {
            "raw_image": image_bytes,
            "prompt": question,
            "prompt_text": question,
            "short_answer": answer,
            "short_answer_text": answer,
            "response_text": f"Predicted Waypoints: {answer}",
        }
```

### 2. File Layout

- **ChartQA:**
  - Format: **ArrayRecord** (`grain.ArrayRecordDataSource`).
  - Shard naming pattern:
    - Train: `human_train@90` (e.g. `/cns/me-d/home/lenck/cms/rs=6.3/array_record/chartqa/human_train@90`)
    - Test/Val: `human_val@10` (e.g. `/cns/me-d/home/lenck/cms/rs=6.3/array_record/chartqa/human_val@10`)
  - Sidecar files: None.

- **Trajectory:**
  - Format: **Bagz** (`grain.BagDataSource`).
  - Shard naming pattern:
    - Train: `wod_e2e_train@100.data.bagz`
    - Test/Val: `wod_e2e_test@20.data.bagz`
  - Sidecar files: None.

### 3. Prompt Template

- **ChartQA:**

```python
"<|turn>user\nInspect the following chart and answer the question precisely.\n<|image|>\n{text}<turn|>\n<|turn>model\n"
```

- **Trajectory** (`_DEFAULT_TRAJECTORY_PROMPT` in `trajectory_data.py`):

```python
_DEFAULT_TRAJECTORY_PROMPT = (
    "<|turn>system\n"
    "Output ONLY exactly 20 predicted future trajectory coordinate pairs"
    " representing the next 5 seconds (sampled at 4Hz at 0.25-second"
    " intervals).\n\n"
    "Format: 'X.XX, Y.YY and X.XX, Y.YY and ...'\n"
    "Example: '1.00, 0.00 and 2.00, 0.00 and 3.00, 0.00 and 4.00, 0.00 and"
    " 5.00, 0.00 and 6.00, 0.00 and 7.00, 0.00 and 8.00, 0.00 and 9.00, 0.00"
    " and 10.00, 0.00 and 11.00, 0.00 and 12.00, 0.00 and 13.00, 0.00 and"
    " 14.00, 0.00 and 15.00, 0.00 and 16.00, 0.00 and 17.00, 0.00 and 18.00,"
    " 0.00 and 19.00, 0.00 and 20.00, 0.00'\n\n"
    "Validation Rules:\n"
    "- Each coordinate pair must consist of exactly two numerical values"
    " (X.XX and Y.YY) separated by a comma.\n"
    "- 'X.XX, Y.YY' is valid, whereas 'X.XX and' is invalid because it lacks"
    " a Y coordinate.\n"
    "- Your output must contain exactly 20 pairs separated by exactly 19 '"
    " and ' joints. Double-check your coordinate count.\n"
    "- Do NOT output any reasoning, thought blocks, or explanations. End"
    " your response with exactly one end-of-turn token immediately after the"
    " 20th coordinate pair.<turn|>\n"
    "<|turn>user\n"
    "<|image|>\n"
    "{text}<turn|>\n"
    "<|turn>model\n"
)
```

### 4. Image Expansion

#### a. ExpandImagePlaceholder class

From `chartqa_data.py`:

```python
@dataclasses.dataclass(kw_only=True, frozen=True)
class ExpandImagePlaceholder(grain.MapTransform):
    """Expands the image placeholder token into multiple soft token placeholders."""

    key: str
    soft_token_count_key: str

    def map(self, element: dict[str, Any]) -> dict[str, Any]:
        tokens = np.array(element[self.key])
        soft_token_count = element[self.soft_token_count_key]

        # Add batch dimension for add_variable_extra_tokens_for_images
        tokens_batched = np.expand_dims(tokens, axis=0)

        expanded_batched = _token_utils.add_variable_extra_tokens_for_images(
            tokens=tokens_batched,
            soft_token_counts=[soft_token_count],
        )

        # Squeeze batch dimension back and store as list
        element[self.key] = expanded_batched[0].tolist()
        return element
```

#### b. add_variable_extra_tokens_for_images function

From `_token_utils.py`:

```python
def add_variable_extra_tokens_for_images(
    tokens: np.ndarray,
    *,
    soft_token_counts: list[int],
) -> np.ndarray:
    special_tokens = _tokenizer.Gemma4Tokenizer.special_tokens
    placeholder_token = special_tokens.IMAGE_PLACEHOLDER
    start_token = special_tokens.START_OF_IMAGE
    end_token = special_tokens.END_OF_IMAGE

    batch_size = tokens.shape[0]
    results = []
    for b in range(batch_size):
        row = tokens[b].tolist()
        expanded = []
        image_idx = 0
        for token in row:
            if token == placeholder_token and image_idx < len(soft_token_counts):
                count = soft_token_counts[image_idx]
                expanded.append(_DOUBLE_NEW_LINE_TOKEN)
                expanded.append(start_token)
                expanded.extend([SOFT_TOKEN_PLACEHOLDER] * count)
                expanded.append(end_token)
                expanded.append(_DOUBLE_NEW_LINE_TOKEN)
                image_idx += 1
            else:
                expanded.append(token)
        results.append(expanded)

    max_len = max(len(r) for r in results)
    padded = np.zeros((batch_size, max_len), dtype=np.int32)
    for b, row in enumerate(results):
        padded[b, : len(row)] = row

    return padded
```

#### c. Expansion sequence and values

- **Sequence:** `[_DOUBLE_NEW_LINE_TOKEN, start_token, *([SOFT_TOKEN_PLACEHOLDER] * count), end_token, _DOUBLE_NEW_LINE_TOKEN]`
- **Value of N:** 280 (max soft tokens).
- **Vision-Tower configuration:** Derived from `output_length=280` configured on the `gemma_vision.VisionEncoder` in `hd_gemma_network.py` (`PrimeDiffusionGemmaModel._ensure_vision_config`):

```python
vision_encoder = gemma_vision.VisionEncoder(
    name="vision_encoder",
    d_model=1152,
    num_layers=27,
    num_heads=16,
    ffw_hidden=4304,
    output_length=280,
    use_clipped_linears=False,
    standardize_embeddings=True,
)
```

### 5. Tokenizer Constants

- **Tokenizer Class:** `gemma.gm.text.Gemma4Tokenizer` (version 4)
- **Vocabulary File:** `/tfhub/prod/ml-gemma/GEMMA-4.0-TOKENIZER/1/gemma4_cleaned_262144.model`
- **Integer IDs:**
  - `PAD`: 0
  - `EOS`: 1
  - `BOS`: 2
  - `START_OF_TURN`: **105** (`<|turn>`)
  - `END_OF_TURN`: **106** (`<turn|>`)
  - `IMAGE_PLACEHOLDER`: **258880** (`<|image|>`)
  - `START_OF_IMAGE`: **255999** (`<|image`)
  - `END_OF_IMAGE`: **258882** (`<image|>`)
  - `BEGIN_OF_TOOL_RESPONSE`: 50
  - `\n\n` Token:
    - Encoded by the tokenizer to ID **203** (or single `\n` to **104**).
    - **Note:** The data pipeline hardcodes `_DOUBLE_NEW_LINE_TOKEN = 108` in `_token_utils.py` for image expansion.

### 6. Target Alignment

#### a. get_no_mm_indices

From `_token_utils.py`:

```python
@typechecked
def get_no_mm_indices(
    *,
    tokens: Int['B L'],  # pyrefly: ignore[not-a-type]
    l_no_mm: int,
    num_tokens_per_image: int,
) -> Int['B L_no_mm']:  # pyrefly: ignore[not-a-type]
    """Get the indices of the tokens which are not MM."""
    del num_tokens_per_image  # Unused, we use dynamic offsets from tokens.
    special_tokens = _tokenizer.Gemma4Tokenizer.special_tokens

    # Dynamic calculation of image token offsets
    actual_soft_token_counts = jnp.sum(tokens == SOFT_TOKEN_PLACEHOLDER, axis=-1)
    actual_offsets = actual_soft_token_counts + 3

    p = jnp.argmax(tokens == special_tokens.START_OF_IMAGE, axis=-1)  # shape (B,)
    has_image = jnp.any(
        tokens == special_tokens.START_OF_IMAGE, axis=-1
    )  # shape (B,)

    j = jnp.arange(l_no_mm)
    j = jnp.broadcast_to(j, (tokens.shape[0], l_no_mm))

    is_after_image = j >= (p[..., None] - 1)
    offset_by_conditional = has_image[..., None] * actual_offsets[..., None]
    new_text_tokens_pos = j + is_after_image * offset_by_conditional
    return new_text_tokens_pos
```

#### b. shift_encoder_targets_for_multimodal

From `sft_model.py`:

```python
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

    Args:
        encoder_target: Original targets, shape [B, L].
        encoder_target_mask: Original target mask, shape [B, L].
        prompt: Prompt tokens, shape [B, PromptLen].
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
```

### 7. Canvas Geometry

- **ChartQA:**
  - `prompt_len = 512`
  - `canvas_size = 256`
  - `num_canvases = 2`
- **Trajectory:**
  - `prompt_len = 512`
  - `canvas_size = 128`
  - `num_canvases = 3`
- **Truncation and Padding Rules:**
  - Prompts are truncated at `prompt_len` and right-padded with `0` (`Gemma4Tokenizer.special_tokens.PAD`) using `gm.data.Pad(key="prompt", max_length=prompt_len, truncate=True)`.
  - Answers are padded with `Gemma4Tokenizer.special_tokens.PAD` (`0`) when split across multiple canvases inside `CanvasChunker`. The empty positions on non-active canvases are masked out using the `canvas_mask`.

---

## Part 2: Evaluation & Sampling Configuration (Q-D5)

### 1. Evaluator Setup

- **Evaluator Class:** `sft_model.GemmaSamplingEvaluator`
- **Dataset Splitting / Number of Examples:**
  - Evaluated on test splits (ChartQA `human_val`, Trajectory `wod_e2e_test`) with a hard slice stop of 256 examples (`slice_stop=256`).
- **Batch Size:**
  - ChartQA: `batch_size = 16`
  - Trajectory: `batch_size = 8`
- **Number of Batches:**
  - ChartQA: `eval_num_batches = None` (runs over the full 256 examples; 16 batches).
  - Trajectory: `eval_num_batches = 4` (evaluates `4 * 8 = 32` examples during training loops).
- **Checkpoint Selection Rule:**
  - Configured with `run=kd.evals.StandaloneEveryCheckpoint()`, which launches evaluation on all checkpoint steps (every 1000 steps). The reported final accuracy is chosen by selecting the checkpoint step that achieves the highest accuracy on the validation set.

### 2. Sampler Constants

The sampling settings are configured inside `make_ar_evals` in `ar_eval.py`:

- `max_denoising_steps`: Evaluated across step configurations 32, 64, and 96 (`AR_DENOISING_STEPS = [32, 64, 96]`).
- `entropy_bound`: Not configured (uses standard `DiffusionSampler` without early stopping by default since `use_early_stopping=False` in configs).
- **Temperature / Annealing:** Uses `DiscreteDDIMStep` with a constant `temperature=0.7` and `logits_dtype=jnp.bfloat16`. No dynamic annealing schedules are used during evaluation.
- `canvas_length`: Matches `canvas_size` of the training configuration (`256` for ChartQA, `128` for Trajectory).
- `num_canvases`: Matches `num_canvases` of the training configuration (`2` for ChartQA, `3` for Trajectory).
- **Early-stop settings:** Disabled (`use_early_stopping=False` is passed from configs).
- **Seed Policy:** Training and evaluations are initialized with a constant global seed: `cfg.seed = 42`. JAX RNG keys are split using streams `default` and `sampling`.

### 3. Metric Definitions

#### a. ChartQA Accuracy (Relaxed Numeric Tolerance)

From `chartqa_eval.py`:

```python
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
```

#### b. Trajectory Metric (ADE / FDE & Waypoint Parsing)

From `trajectory_eval.py`:

```python
def parse_trajectory(text: str) -> np.ndarray:
    """Parses coordinates from text into a 2D array of floats.

    Expected format: "Predicted Waypoints: x1, y1 and x2, y2 ..." or just
    coordinates. It extracts all numbers matching float pattern and reshapes to
    (-1, 2).
    """
    if not isinstance(text, str):
        text = str(text)

    # Try to extract only the waypoints part if prefix is present
    match = re.search(r"(?i)Predicted Waypoints:\s*(.*)", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Find all floats
    pattern = r"-?\d+\.\d+"
    floats = re.findall(pattern, text)

    if not floats:
        return np.empty((0, 2), dtype=np.float32)

    num_coords = len(floats)
    valid_len = (num_coords // 2) * 2
    coords = np.array(floats[:valid_len], dtype=np.float32).reshape(-1, 2)
    return coords


def pad_or_truncate(traj: np.ndarray, target_len: int) -> np.ndarray:
    """Pads or truncates trajectory to target_len.

    If traj is empty, returns array of zeros.
    If padding, repeats the last valid point.
    """
    if traj.shape[0] == 0:
        return np.zeros((target_len, 2), dtype=np.float32)

    curr_len = traj.shape[0]
    if curr_len == target_len:
        return traj
    elif curr_len > target_len:
        return traj[:target_len]
    else:
        # Pad by repeating the last element
        num_pad = target_len - curr_len
        last_elem = traj[-1:]
        padding = np.repeat(last_elem, num_pad, axis=0)
        return np.concatenate([traj, padding], axis=0)


@dataclasses.dataclass(kw_only=True, frozen=True)
class TrajectoryADE(TrajectoryMetricBase):
    """Computes Average Displacement Error (ADE) on trajectory coordinates."""

    def score_example(self, generated_text: str, ground_truth_text: str) -> float:
        gt_traj = parse_trajectory(ground_truth_text)
        pred_traj = parse_trajectory(generated_text)

        if gt_traj.shape[0] == 0:
            return 0.0

        target_len = gt_traj.shape[0]
        pred_traj = pad_or_truncate(pred_traj, target_len)

        dists = np.linalg.norm(pred_traj - gt_traj, axis=1)
        return float(np.mean(dists))


@dataclasses.dataclass(kw_only=True, frozen=True)
class TrajectoryFDE(TrajectoryMetricBase):
    """Computes Final Displacement Error (FDE) on trajectory coordinates."""

    def score_example(self, generated_text: str, ground_truth_text: str) -> float:
        gt_traj = parse_trajectory(ground_truth_text)
        pred_traj = parse_trajectory(generated_text)

        if gt_traj.shape[0] == 0:
            return 0.0

        target_len = gt_traj.shape[0]
        pred_traj = pad_or_truncate(pred_traj, target_len)

        dists = np.linalg.norm(pred_traj - gt_traj, axis=1)
        return float(dists[-1])
```
