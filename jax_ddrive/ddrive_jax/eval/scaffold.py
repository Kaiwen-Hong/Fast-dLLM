"""Deep-JSON scaffold construction for Fast-dDrive generation (the pre-loop state of
`mdm_sample_deep_scaffold`).  Runs in the PyTorch (`ddrive`) env because section_utils
uses torch — but loads NO model weights (pure tokenizer/section layout).  Produces the
masked sequence x_t0 = [prompt | scaffold(value slots=mask_id)] and response_block_idx
(rbi), reusing the same `section_utils` functions the training prep used (loss parity
7.9e-8).  Mirrors generation_utils.py L114-202.

Also provides `messages_from_prompt` so the capture/prep build the processor inputs the
EXACT same way as fast_ddrive/eval/batch_inference.py (single `<image>` placeholder →
all images inserted there)."""
import sys
import numpy as np

MASK_ID = 151665
NULL_ID = 151666
IM_END = 151645


def messages_from_prompt(prompt: str, images: list):
    """Replicate batch_inference.generate's content construction EXACTLY (running image
    cursor: each non-last <image> consumes one image, the last dumps the remaining ones).
    For the canonical 1-placeholder/3-image Waymo prompt this yields [img0,img1,img2]."""
    content = []
    image_idx = 0
    if "<image>" in prompt:
        parts = prompt.split("<image>")
        for idx, part in enumerate(parts):
            if part:
                content.append({"type": "text", "text": part})
            if idx < len(parts) - 1:
                if idx == len(parts) - 2:                # last placeholder → remaining images
                    while image_idx < len(images):
                        content.append({"type": "image", "image": images[image_idx]})
                        image_idx += 1
                elif image_idx < len(images):            # earlier placeholder → one image
                    content.append({"type": "image", "image": images[image_idx]})
                    image_idx += 1
    else:
        content.append({"type": "text", "text": prompt})
        while image_idx < len(images):
            content.append({"type": "image", "image": images[image_idx]})
            image_idx += 1
    return [{"role": "user", "content": content}]


def build_scaffold(prompt_ids, tokenizer, snapshot_dir: str,
                   mask_id: int = MASK_ID, null_id: int = NULL_ID, block_size: int = 32):
    """prompt_ids: 1D int (the chat-templated prompt, add_generation_prompt=True, with the
    image placeholders already expanded by the processor).  Returns:
        x_t0 [L] int64  = [prompt | scaffold]  (scaffold value slots = mask_id)
        rbi  [L] int64  = response_block_idx (prompt = -1)
        orig_len int    = len(prompt_ids)
        block_to_section dict
    """
    import math
    if snapshot_dir not in sys.path:
        sys.path.insert(0, snapshot_dir)
    from section_utils import build_deep_json_scaffold, SECTION_KEYS
    scaffold_tokens, section_ranges, scaffold_mask_list = build_deep_json_scaffold(
        tokenizer, mask_id=mask_id, null_id=null_id,
        explanation_block_size=32, explanation_max_blocks=6)
    prompt_ids = [int(t) for t in np.asarray(prompt_ids).reshape(-1)]
    x_t0 = prompt_ids + [int(t) for t in scaffold_tokens]
    orig_len = len(prompt_ids)
    seqlen = len(x_t0)

    # ── response_block_idx: EXACT replica of generation_utils.py L134-202 (the inference
    #    block layout used by mdm_sample_deep_scaffold), not the training function. ──
    rbi = np.full(seqlen, -1, dtype=np.int64)
    current_block = 0
    assigned = set()
    b2s = {}
    for section_name in SECTION_KEYS:
        if section_name not in section_ranges:
            continue
        sec_start, sec_end = section_ranges[section_name]
        value_positions = [orig_len + i for i in range(sec_start, sec_end)
                           if not scaffold_mask_list[i]]
        if not value_positions:
            current_block += 1
            continue
        n_steps = max(1, math.ceil(len(value_positions) / block_size))
        for blk in range(current_block, current_block + n_steps):
            b2s[blk] = section_name
        for vi, abs_pos in enumerate(value_positions):
            rbi[abs_pos] = current_block + min(vi // block_size, n_steps - 1)
            assigned.add(abs_pos)
        for i in range(sec_start, sec_end):
            abs_pos = orig_len + i
            if scaffold_mask_list[i] and abs_pos not in assigned:
                best = -1
                for delta in range(1, sec_end - sec_start + 10):
                    for cand in (abs_pos + delta, abs_pos - delta):
                        if cand in assigned:
                            best = int(rbi[cand])
                            break
                    if best >= 0:
                        break
                if best >= 0:
                    rbi[abs_pos] = best
                    assigned.add(abs_pos)
        current_block += n_steps
    for i in range(len(scaffold_tokens)):
        abs_pos = orig_len + i
        if abs_pos not in assigned:
            best = -1
            for delta in range(1, seqlen):
                for cand in (abs_pos + delta, abs_pos - delta):
                    if 0 <= cand < seqlen and cand in assigned:
                        best = int(rbi[cand])
                        break
                if best >= 0:
                    break
            if best >= 0:
                rbi[abs_pos] = best
                assigned.add(abs_pos)
    return np.array(x_t0, np.int64), rbi, orig_len, b2s
