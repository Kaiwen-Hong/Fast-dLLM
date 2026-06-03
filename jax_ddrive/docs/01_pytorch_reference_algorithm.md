# Fast-dDrive — PyTorch Reference Algorithm (porting bible)

Extracted verbatim-with-line-refs from the HF `trust_remote_code` source cached at
`/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/<hash>/`
(`modeling.py` 3120 L, `section_utils.py` 803 L, `generation_utils.py` 1192 L).
This is the ground truth the JAX port must reproduce. Section numbers map to the 6 port areas.

> Milestone-critical: **§1 (loss) + §2 (mask) + §3 (scaffold)**. §4 M-RoPE is a no-op for text-only
> training. §5 decoders are deferred (inference). §6 is the weight-conversion key map.

## 1. Training forward + SASD loss  (the loss-decrease milestone)

- Sequence is **doubled into `[noisy | clean]`**; loss logits come from the **noisy half**:
  `mdm_hidden_states = hidden_states[:, :L//2, :]` then `lm_head(...)` — `modeling.py:2712-2715`.
- Section-weighted CE — `modeling.py:2123-2157` (`compute_section_weighted_loss`):
  ```python
  shift_logits = logits[..., :-1, :]; shift_labels = labels[..., 1:]; shift_weights = weight[..., 1:]
  per_token = CrossEntropyLoss(reduction='none', ignore_index=-100)(shift_logits.view(-1,V), shift_labels.view(-1))
  weighted = per_token * shift_weights.view(-1)
  loss = weighted.sum() / (num_items_in_batch or (shift_labels!=-100).sum())
  ```
  **No explicit 1/t term** (differs from reference `mdlm_loss`). Section weights {critical_objects 1.5,
  explanation 1.0, future_meta_behavior 2.0, trajectory 3.0} applied per-token via block→section map —
  `_build_section_weight_tensor` `modeling.py:2104-2121`.
- **Complementary loss**: a second CE on the clean/causal `causal_logits` over complementary tokens,
  added to the mdm loss — `modeling.py:2764-2766`. Port must include both terms.
- **Per-section Beta noise** — `_sample_section_aware_noise` `modeling.py:2085-2102`:
  `t[block] = Beta(α,β).sample()` per section (schedule string "α,β"), else `U(0,1)`.
  `p_mask_per_block = (1-eps)*t + eps`, eps=`minimum_noise_level`=1e-3 — `modeling.py:2273`.
  Tokens masked by `rand < p_mask[block]` — `:2275-2279`; then **scaffold tokens un-masked**
  `mask &= ~scaffold_mask` — `:2281-2283`. Labels ignore-index = -100 for prompt + unmasked.

## 2. Attention masking — hybrid block-causal on the doubled sequence

`hybrid_block_causal_mask_multiturn` — `modeling.py:178-234`. Operates on `[noisy(0..n) | clean(n..2n)]`,
indexed back to original positions via `response_block_idx` (block id per pos, -1=prompt) and `turn_idx`:
- block-diagonal: both noisy, same turn → bidirectional within block;
- offset-block-causal: noisy response attends clean tokens of prior turns;
- x0-causal: clean half is standard causal.
Selected in forward at `modeling.py:2349-2357` (deep scaffold ⇒ hybrid). Inference variant
`eval_hybrid_block_causal_mask` `:248-279` (prompt causal | response sees prompt | response block-causal).

## 3. Deep JSON scaffold / `section_utils.py`

- `build_deep_json_scaffold` `section_utils.py:109-293`: builds the JSON with placeholder values, tokenizes
  the whole string once, then **boundary pattern-matches** to mark structural tokens as scaffold (frozen).
- Sections in causal order via `boundary_order` `:420-491`: critical_objects → explanation →
  future_meta_behavior → trajectory. Sub-key freezing: critical_objects `:493-520`, fmb `:538-579`,
  trajectory `:584-647` (freezes `[[`, `],`, `,`, `]]"`; coordinate value tokens left maskable).
- `<|NULL|>` = id **151666**; explanation padded to budget (default 192 = 32*6) `:205-237`; stripped at
  inference `generation_utils.py:451-457`. `|<MASK>|` id ≈ 151665.
- Returns `response_block_idx[i]`, `block_to_section[b]`, `scaffold_mask[i]` — consumed by §1/§2.

## 4. M-RoPE  (no-op for text-only)

`apply_multimodal_rotary_pos_emb` `modeling.py:817-859` with `mrope_section=[16,24,24]`. **Training text-only**
sets `position_ids = arange(L).view(1,1,-1).expand(3,B,-1)` — `:1566-1590` ⇒ all 3 sections identical ⇒
plain RoPE. Multimodal builds 3D (t,h,w) ids from `image_grid_thw` `:1456-1556`.

## 5. Three decoders (deferred — inference)

- `mdm_sample_deep_scaffold` (SD) `generation_utils.py:58-470`: build scaffold w/ MASK at value positions;
  iterate block-by-block, unmask where `softmax(logit)[x1] > threshold` (0.9), KV-cache per finalized block.
- `scaffold_speculative_sample` (SS, canonical) `:479-749`: prefill prompt; per block draft (block-diff
  attn) + AR verify (causal); accept run of matching/scaffold tokens; crop cache, advance.
- `scaffold_spec_with_ss_multi_traj` `:758-1164`: deterministic SS for sections 1-3, fork KV N times,
  stochastic SS on trajectory per fork, weighted-merge waypoints.

## 6. Parameter tree → safetensors keys (for `a2d.py`-style conversion)

```
Fast_dDriveForConditionalGeneration
├── model.visual.{patch_embed, rotary_pos_emb, blocks.N.{norm1,attn.qkv,attn.proj,norm2,mlp...}, merger}
├── model.language_model.embed_tokens
├── model.language_model.layers.N.self_attn.{q,k,v,o}_proj  (+ rotary_emb.inv_freq)
├── model.language_model.layers.N.mlp.{gate,up,down}_proj
├── model.language_model.layers.N.{input_layernorm, post_attention_layernorm}
├── model.language_model.norm
└── lm_head  (TIED to embed_tokens)
```
Conversion regex `modeling.py:1862-1865`: `^visual→model.visual`, `^model(?!\.(language_model|visual))→model.language_model`.
PyTorch Linear (out,in) → JAX kernel (in,out) transpose; norms/embeddings no transpose (per `a2d.py`).
