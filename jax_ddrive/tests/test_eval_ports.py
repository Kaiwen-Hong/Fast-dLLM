"""CPU regression tests for the eval-pipeline ports (no torch / no GPU / no proto):
  * messages_from_prompt — exactly len(images) image blocks for any placeholder count,
    in running-cursor order (the audit fix vs the PyTorch reference).
  * get_rope_index_numpy — text-only == plain arange on all 3 channels; single-image case
    matches the Qwen2.5-VL 3D M-RoPE structure (text-before / image grid / text-after).
Run: JAX_PLATFORMS=cpu python jax_ddrive/tests/test_eval_ports.py"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.eval.scaffold import messages_from_prompt
from ddrive_jax.eval.rope_index import get_rope_index_numpy

VSTART, IMG = 151652, 151655


def _imgs(prompt, images):
    c = messages_from_prompt(prompt, images)[0]["content"]
    return [x["image"] for x in c if x["type"] == "image"]


def test_messages_counts():
    # canonical: 1 placeholder, 3 images -> all 3 in order
    assert _imgs("AA <image> BB", ["i0", "i1", "i2"]) == ["i0", "i1", "i2"]
    # >=2 placeholders must NOT duplicate (the audit fix)
    assert _imgs("<image>a<image>b<image>c", ["i0", "i1", "i2"]) == ["i0", "i1", "i2"]
    assert _imgs("<image>x<image>", ["i0", "i1"]) == ["i0", "i1"]
    # no placeholder -> all appended after the text
    assert _imgs("just text", ["i0", "i1", "i2"]) == ["i0", "i1", "i2"]
    # more placeholders than images -> never index past the end (no IndexError)
    assert _imgs("<image><image><image>", ["i0"]) == ["i0"]
    print("[ok] test_messages_counts")


def test_rope_text_only():
    ids = np.array([10, 11, 12, 13, 14], np.int64)
    pos = get_rope_index_numpy(ids, None)
    assert pos.shape == (3, 5)
    for ch in range(3):
        assert (pos[ch] == np.arange(5)).all(), pos
    print("[ok] test_rope_text_only")


def test_rope_single_image():
    # [t0, t1, <vstart>, IMG,IMG,IMG,IMG, t2]  with grid (t=1,h=4,w=4) -> llm 2x2 = 4 img tokens
    ids = np.array([10, 11, VSTART, IMG, IMG, IMG, IMG, 12], np.int64)
    grid = np.array([[1, 4, 4]], np.int64)
    pos = get_rope_index_numpy(ids, grid)
    assert pos.shape == (3, 8)
    # text before the FIRST image token (idx 0..2: t0,t1,vstart) -> arange, all channels equal
    assert (pos[:, 0] == 0).all() and (pos[:, 1] == 1).all() and (pos[:, 2] == 2).all()
    # image block at positions 3..6: base = img_start(=3); t const, h=row, w=col (2x2)
    base = 3
    t_img, h_img, w_img = pos[0, 3:7], pos[1, 3:7], pos[2, 3:7]
    assert (t_img == base).all(), t_img                          # temporal constant (t=1, sec/grid=0)
    assert h_img.tolist() == [base + 0, base + 0, base + 1, base + 1], h_img   # rows
    assert w_img.tolist() == [base + 0, base + 1, base + 0, base + 1], w_img   # cols
    # text after image (pos 7) continues at max(image)+1 = base + max(llm_h,llm_w) = 3+2 = 5
    assert (pos[:, 7] == 5).all(), pos[:, 7]
    print("[ok] test_rope_single_image")


if __name__ == "__main__":
    test_messages_counts()
    test_rope_text_only()
    test_rope_single_image()
    print("\nALL EVAL PORT TESTS PASS")
