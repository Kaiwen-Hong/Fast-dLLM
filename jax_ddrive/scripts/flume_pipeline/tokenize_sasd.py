"""Job ② — `TokenizeSASD`: the train-time Grain `MapTransform` (doc 09 §4.4).

This IS `jax_ddrive/eval/prep_train_jax.py` repackaged from an *offline batch script*
into a *per-sample, train-time transform*. It takes one raw msgpack record (Job ① output:
raw prompt/target strings + 3 JPEG blobs) and emits the 13-field SASD tensor dict that the
JAX/TPU training step consumes (doc 09 §1).

Two intentional differences from the oracle (everything else is byte-for-byte identical,
which the parity gate enforces):

  1. INPUT SOURCE — fields come from a msgpack record (`prompt_text`, `target_text`,
     `images[3]` raw JPEG) instead of a JSON item + on-disk image paths. msgpack preserves
     the strings exactly and the JPEG bytes are the same bytes, so tokenization is identical.

  2. CONFIG, NOT DATA — `weight_vec` / `block_alpha` / `block_beta` are expanded HERE at
     train-time from `section_w` / `noise_sched` (injected config; defaults == the oracle's
     `SECTION_W` / `NOISE_SCHED`) rather than being frozen into every dataset row. This is
     the "don't bake config into data" fix (doc 09 §2.4). Same defaults -> same numbers.

To keep a SINGLE source of truth (and guarantee parity), this reuses the oracle's own leaf
functions — `process_gpt`, `compute_section_block_idx_deep_static`,
`build_deep_scaffold_sequences`, `get_rope_index_numpy`, `messages_from_prompt`, and the
token-id constants — rather than reimplementing them. In google3 these would be factored
into a shared lib that both the (retired) oracle and this transform import.

Runs on the oracle's EXACT stack: `ddrive` env (torch 2.11 + transformers 4.57.1 + the
trust_remote_code Qwen2.5-VL modeling + `section_utils`). `grain` is an optional import —
the `MapTransform` base class is cosmetic; the load-bearing google3 wiring lives in
`grain_loader.py`.
"""
import io
import os
import sys

import numpy as np

# ── resolve repo + snapshot exactly like prep_train_jax.py so the same modules import ──
SNAP = os.environ.get(
    "FASTDDRIVE_SNAP",
    "/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
    "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
REPO = os.environ.get("FASTDDRIVE_REPO", "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, REPO, os.path.join(REPO, "eval"), SNAP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Train-time SASD config (doc 09 §2.4: these are CONFIG knobs, NOT per-row data). Defaults
# are byte-identical to prep_train_jax.SECTION_W / NOISE_SCHED so parity holds out of the box;
# override at construction to retune weights/noise WITHOUT re-running the ETL.
DEFAULT_SECTION_W = {"critical_objects": 1.5, "explanation": 1.0,
                     "future_meta_behavior": 2.0, "trajectory": 3.0}
DEFAULT_NOISE_SCHED = {"critical_objects": (1.0, 2.0), "explanation": (1.0, 1.0),
                       "future_meta_behavior": (1.0, 1.5), "trajectory": (2.0, 1.0)}

# Label-span scan sentinels (Qwen chat <|im_start|>assistant\n = 151644, 77091, 198), and the
# extra vision token (151654) that joins {IMAGE_TOK, VSTART} in the vision mask. Mirrors oracle.
_ASSIST_START = (151644, 77091, 198)
_VISION_EXTRA = 151654

try:                                            # grain base class is cosmetic; degrade gracefully
    import grain.python as grain
    _BASE = grain.MapTransform
except Exception:                               # noqa: BLE001 - any import failure -> plain class
    grain = None
    _BASE = object


class TokenizeSASD(_BASE):
    """Decode one msgpack record -> 13-field SASD tensor dict (= prep_train_jax npz row).

    Build once (loads the Qwen processor/tokenizer + the deep-scaffold sequences), then call
    `.map(record_bytes)` per sample. `.map` accepts either raw msgpack `bytes` (Grain path) or
    an already-unpacked record `dict` (parity-test path)."""

    def __init__(self, model_dir=SNAP, section_w=None, noise_sched=None,
                 min_pixels=200704, max_pixels=200704):
        from transformers import AutoProcessor, AutoTokenizer
        from section_utils import build_deep_scaffold_sequences

        self.section_w = dict(section_w) if section_w else dict(DEFAULT_SECTION_W)
        self.noise_sched = dict(noise_sched) if noise_sched else dict(DEFAULT_NOISE_SCHED)

        # EXACT replica of prep_train_jax.main() processor/tokenizer setup (lines 84-89):
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.proc = AutoProcessor.from_pretrained(model_dir, use_fast=False)
        self.proc.tokenizer = self.tok
        self.proc.image_processor.min_pixels = min_pixels      # 200704 -> ~720 img tokens (doc 07)
        self.proc.image_processor.max_pixels = max_pixels
        self.seqs = build_deep_scaffold_sequences(self.tok)    # cached once (oracle line 89)

    # ---- Grain entrypoint -------------------------------------------------------------
    def map(self, record):
        from etl_record import unpack
        rec = unpack(record) if isinstance(record, (bytes, bytearray)) else record
        return self._transform(rec)

    __call__ = map                              # convenience for non-grain callers

    # ---- the actual prep_train_jax body, per sample -----------------------------------
    def _transform(self, rec):
        import torch
        from PIL import Image
        from prep_train_jax import process_gpt, MASK_ID, IM_END, BD, IMAGE_TOK, VSTART
        from section_utils import compute_section_block_idx_deep_static
        from ddrive_jax.eval.scaffold import messages_from_prompt
        from ddrive_jax.eval.rope_index import get_rope_index_numpy

        # 1-3. normalize target + build the chat messages (oracle lines 97-103) -----------
        human = rec["prompt_text"]
        gpt = process_gpt(rec["target_text"], self.tok)
        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in rec["images"]]
        user_content = messages_from_prompt(human, imgs)[0]["content"]
        msgs = [{"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": gpt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

        # 4. Qwen processor (mirror oracle line 104-107: return_tensors='pt' then float16) --
        inputs = self.proc(text=[text], images=imgs, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        pv = inputs["pixel_values"].float().cpu().numpy().astype(np.float16)
        thw = inputs["image_grid_thw"].cpu().numpy().astype(np.int64)

        # 5. pad to BD multiple with MASK_ID (oracle 108-110) ------------------------------
        if len(ids) % BD:
            ids += [MASK_ID] * (BD - len(ids) % BD)
        L = len(ids)

        # 6. labels: response span after <|im_start|>assistant\n, end at <|im_end|> +2 -----
        labels = [-100] * L
        for i in range(L - 2):
            if (ids[i], ids[i + 1], ids[i + 2]) == _ASSIST_START:
                rs = i + 3
                re_ = rs
                while re_ < L and ids[re_] != IM_END:
                    re_ += 1
                re_ = min(re_ + 2, L)            # include <|im_end|> AND trailing \n (oracle 120)
                for j in range(rs, re_):
                    labels[j] = ids[j]
                break

        # 7. SASD block/section structure (oracle 124-125) --------------------------------
        ii = torch.tensor(ids)[None]
        ll = torch.tensor(labels)[None]
        rbi, turn, n_blocks, scaff, b2s = compute_section_block_idx_deep_static(ll, ii, self.seqs, BD)
        rbi = rbi.view(-1).numpy().astype(np.int32)
        turn = turn.view(-1).numpy().astype(np.int32)
        scaff = scaff.view(-1).numpy().astype(bool)

        # 8. TRAIN-TIME expansion of weight/noise from CONFIG (was baked into data) --------
        wv = np.ones(L, np.float32)
        ba = np.ones(n_blocks, np.float32)
        bb = np.ones(n_blocks, np.float32)
        for i, b in enumerate(rbi):
            if b >= 0 and int(b) in b2s and b2s[int(b)] in self.section_w:
                wv[i] = self.section_w[b2s[int(b)]]
        for b in range(n_blocks):
            if b2s.get(b) in self.noise_sched:
                ba[b], bb[b] = self.noise_sched[b2s[b]]

        # 9. rope / vision mask (oracle 135-136) ------------------------------------------
        pos = get_rope_index_numpy(np.array(ids, np.int64), thw).astype(np.int32)
        vmask = np.array([(t == IMAGE_TOK) or (t == VSTART) or (t == _VISION_EXTRA)
                          for t in ids], bool)

        # 10. the 13-field tensor contract (doc 09 §1) ------------------------------------
        return {
            "input_ids": np.array(ids, np.int64),
            "labels": np.array(labels, np.int64),
            "rbi": rbi,
            "turn": turn,
            "scaffold": scaff,
            "weight_vec": wv,
            "block_alpha": ba,
            "block_beta": bb,
            "n_blocks": np.int64(n_blocks),
            "pixel_values": pv,
            "image_grid_thw": thw,
            "position_ids": pos,
            "vision_mask": vmask,
        }
