"""Phase-4 gate: JAX Qwen2.5-VL vision tower vs PyTorch oracle (capture_oracle_vit.py).
Gate: image_embeds rel-max < 1e-2 (ViT looser due to many ops; aim < 1e-3)."""
import sys
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.models.vision_qwen25vl import VisionConfig
from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_vit

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
ORACLE = "/home/kaiwen/data/fast-ddrive/ref_logits/vit_oracle.npz"


def main():
    d = np.load(ORACLE)
    pv = jnp.asarray(d["pixel_values"], dtype=jnp.float32)
    thw = d["image_grid_thw"]
    ref = d["image_embeds"]
    print("pixel_values", pv.shape, "grid", thw.tolist(), "ref embeds", ref.shape)

    vit, info = load_fast_ddrive_vit(SNAP, VisionConfig(dtype=jnp.float32), dtype=jnp.float32)
    out = np.asarray(vit(pv, thw), dtype=np.float32)
    relmax = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9))
    rell2 = float(np.linalg.norm((out - ref).ravel()) / (np.linalg.norm(ref.ravel()) + 1e-9))
    print(f"\n=== PHASE 4 ViT PARITY ===  shape {out.shape}")
    print(f"end-to-end image_embeds rel-max {relmax:.3e}  rel-L2 {rell2:.3e}")

    # Isolate the math from the cuDNN-Conv3d seed: feed PyTorch's patch-embed output
    # (vit_debug.npz) into our blocks+merger. This is the true architecture parity.
    iso = None
    try:
        from ddrive_jax.models.vision_qwen25vl import get_window_index, cu_seqlens_full, _seg_ids_from_cu
        dd = np.load("/home/kaiwen/data/fast-ddrive/ref_logits/vit_debug.npz")
        cfg = vit.cfg; u = cfg.spatial_merge_unit; N = int(pv.shape[0])
        wi, cuw = get_window_index(thw, cfg); cu = cu_seqlens_full(thw)
        cos, sin = vit._rotary(thw, wi)
        # direct patch-embed (Conv3D-as-linear reshape) gate: our matmul vs PyTorch's cuDNN conv.
        pe = np.asarray(vit.patch_embed(pv)).reshape(N // u, u, -1)[wi].reshape(N, -1)
        pe_rel = float(np.abs(pe - dd["patch_reord"]).max() / (np.abs(dd["patch_reord"]).max() + 1e-9))
        print(f"patch_embed (conv-as-linear) rel-max {pe_rel:.3e}  (~6.5e-4 cuDNN-conv seed; >5e-3 ⇒ reshape bug)")
        x = jnp.asarray(dd["patch_reord"])
        sf = jnp.asarray(_seg_ids_from_cu(cu, N)); mf = (sf[:, None] == sf[None, :])
        sw = jnp.asarray(_seg_ids_from_cu(cuw, N)); mw = (sw[:, None] == sw[None, :])
        for i, blk in enumerate(vit.blocks):
            x = blk(x, cos, sin, mf if i in cfg.fullatt_block_indexes else mw)
        x = np.asarray(vit.merger(x))[np.argsort(wi)]
        iso = float(np.abs(x - ref).max() / (np.abs(ref).max() + 1e-9))
        print(f"blocks+merger on PyTorch patch-embed rel-max {iso:.3e}  (cuDNN-conv seed removed)")
    except FileNotFoundError:
        pe_rel = None
        print("(run scripts/debug_vit.py first for the isolated metric)")

    # Pass requires BOTH: blocks+merger arch parity (iso<1e-3) AND patch_embed reshape correct
    # (pe_rel<5e-3 catches reshape/transpose bugs while tolerating the ~6.5e-4 cuDNN seed).
    gate = ((iso is not None and iso < 1e-3 and (pe_rel is None or pe_rel < 5e-3))
            or (iso is None and relmax < 1e-2))
    print(f"\nPHASE4_VIT_{'PASS' if gate else 'FAIL'} "
          f"(arch parity {iso if iso is not None else relmax:.3e}; end-to-end {relmax:.3e} carries the cuDNN-conv seed)")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
