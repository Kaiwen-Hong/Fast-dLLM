# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Single source of truth for the ChartQA/trajectory data-pipeline constants.

Everything here mirrors the INTERNAL SFT stack verbatim (provenance:
``discreteGemma/response-from-jestki.md`` Doc 2 Part 1 — the internal agent's
Q-D3 answer). If the internal schema changes, this is the one file to update
(design doc §5 B4).
"""

# ---- tf.train.Example feature keys [internal·reply §1b] ----
FEATURE_IMAGE = "image/encoded"  # bytes_list: JPEG/PNG-encoded original image
FEATURE_QUESTION = "question"    # bytes_list: utf-8
FEATURE_ANSWER = "answer"        # bytes_list: utf-8

# ---- response templates [internal·reply §1c] (LOAD-BEARING: the ChartQA ----
# ---- metric extracts the answer via the "The answer is:" prefix)        ----
CHARTQA_RESPONSE_TEMPLATE = "The answer is: {answer}"
TRAJECTORY_RESPONSE_TEMPLATE = "Predicted Waypoints: {answer}"

# ---- prompt templates [internal·reply §3] ----
CHARTQA_PROMPT_TEMPLATE = (
    "<|turn>user\nInspect the following chart and answer the question"
    " precisely.\n<|image|>\n{text}<turn|>\n<|turn>model\n"
)

# Kept for the future WOD/trajectory twin (verbatim internal).
TRAJECTORY_PROMPT_TEMPLATE = (
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

# ---- canvas geometry [internal·reply §7] ----
CHARTQA_PROMPT_LEN = 512
CHARTQA_CANVAS_SIZE = 256
CHARTQA_NUM_CANVASES = 2
TRAJECTORY_PROMPT_LEN = 512
TRAJECTORY_CANVAS_SIZE = 128
TRAJECTORY_NUM_CANVASES = 3

# ---- local artifact layout (mirrors internal human_train@90 / human_val@10;
# ---- ChartQA = ArrayRecord, toy/trajectory = Bagz [internal·reply §2]) ----
LOCAL_DATA_ROOT = "/home/kaiwen/data/dgemma_e2b/chartqa"
TRAIN_PATTERN = "human_train-{i:05d}-of-{n:05d}.arrayrecord"
VAL_PATTERN = "human_val-{i:05d}-of-{n:05d}.arrayrecord"
TOY_BAGZ = "chartqa_toy.bagz"

# ---- E2B vision preprocessing (from Gemma4_E2B.config.vision_encoder,   ----
# ---- verified at runtime 2026-07-07: patch 16 / pooling 3 / budget 280) ----
E2B_PATCH_SIZE = 16
E2B_POOLING_KERNEL_SIZE = 3
E2B_MAX_SOFT_TOKENS = 280
# Fixed-square input => deterministic actual soft-token count:
# floor(sqrt(280))**2 = 256 (< 280 => padded patch rows always exist — the
# C6(c) padding-invariance test relies on this).
E2B_N_SOFT = 256
E2B_IMAGE_SIZE = 768  # square resize side (48x48 patches of 16px, pooled 3x)
