"""Pure-numpy port of Qwen2.5-VL `get_rope_index` (the 3D M-RoPE position builder),
matching modeling.py L1456-1557 for the image branch (no video). Single batch (B=1).

For an image token at merged-grid (gt, gh, gw): (t,h,w) = (base, base+gh, base+gw)
with base = st_idx + text_len_before_this_image, and second_per_grid_t=0 for images
(so the temporal channel is constant within an image). Text tokens (incl. the entire
JSON scaffold/response) get contiguous 1D positions identical across the 3 channels.

Validated to be bit-identical to `model.model.get_rope_index(...)` (see
scripts/capture_oracle_sd_mm.py + parity check)."""
import numpy as np


def get_rope_index_numpy(
    input_ids,
    image_grid_thw,
    *,
    spatial_merge_size: int = 2,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    vision_start_token_id: int = 151652,
):
    """input_ids: 1D int array [S].  image_grid_thw: [n_img, 3] of (t,h,w) PATCH counts.
    Returns position_ids int64 [3, S] (channels = temporal, height, width)."""
    tokens = [int(t) for t in np.asarray(input_ids).reshape(-1)]
    S = len(tokens)
    pos = np.ones((3, S), dtype=np.int64)
    if image_grid_thw is None or len(image_grid_thw) == 0:
        # pure text: identical 1D positions on all 3 channels
        pos[:] = np.arange(S, dtype=np.int64)[None, :]
        return pos

    grid = np.asarray(image_grid_thw).reshape(-1, 3)
    n_img = grid.shape[0]
    llm_pos_list = []
    st = 0
    for img_i in range(n_img):
        ed = tokens.index(image_token_id, st)               # first placeholder at/after st
        t, h, w = int(grid[img_i, 0]), int(grid[img_i, 1]), int(grid[img_i, 2])
        llm_t, llm_h, llm_w = t, h // spatial_merge_size, w // spatial_merge_size
        text_len = ed - st
        st_idx = int(llm_pos_list[-1].max()) + 1 if llm_pos_list else 0
        if text_len > 0:
            llm_pos_list.append(np.broadcast_to(np.arange(text_len) + st_idx, (3, text_len)).copy())
        base = text_len + st_idx
        # second_per_grid_t = 0 for images -> temporal index is all-zero before offset
        t_index = np.zeros(llm_t * llm_h * llm_w, dtype=np.int64)
        h_index = np.broadcast_to(np.arange(llm_h).reshape(1, -1, 1), (llm_t, llm_h, llm_w)).reshape(-1)
        w_index = np.broadcast_to(np.arange(llm_w).reshape(1, 1, -1), (llm_t, llm_h, llm_w)).reshape(-1)
        llm_pos_list.append(np.stack([t_index, h_index, w_index]) + base)
        st = ed + llm_t * llm_h * llm_w
    if st < S:
        st_idx = int(llm_pos_list[-1].max()) + 1 if llm_pos_list else 0
        text_len = S - st
        llm_pos_list.append(np.broadcast_to(np.arange(text_len) + st_idx, (3, text_len)).copy())
    llm_positions = np.concatenate(llm_pos_list, axis=1).reshape(3, -1)
    pos[:, : llm_positions.shape[1]] = llm_positions
    return pos
