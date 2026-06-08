"""Path-A FSDP training driver for Fast-dDrive (proxy on CPU 8-device emulation; real on TPU).

The model is config-driven:
  --proxy : a SMALL Qwen25TextConfig (few layers, small hidden) but FULL vocab, with dummy
            FROZEN image embeds (stop_gradient(random)) -- for CPU sharding/optim/ckpt
            verification. The SAME code paths run the full model.
  real    : load_fast_ddrive text + frozen ViT (GPU/TPU only; not run on CPU here).

Mechanics (identical for proxy + real):
  - build a ('fsdp','tp') mesh; pure-FSDP (n_tp=1) here.
  - device_put params onto NamedSharding from sharding.param_pspecs (embedding replicated,
    largest-axis kernels sharded on 'fsdp').
  - init optimizer state already sharded (same fsdp rule, embedding-shaped leaf replicated).
  - jit train_step with in/out_shardings: data sharded P('fsdp', ...) on the leading B axis,
    params/opt keep their sharding, loss/aux replicated.
  - per-sample forward = the reference single-sample SASD step, vmapped over B (each sample
    uses ITS OWN cos/sin/mask4d/img_pos/image_embeds).
  - GLOBAL DP-correct loss = sum_b(section_sum + causal_sum) / sum_b(num_items).
  - cos/sin/mask4d/img_pos are host-precomputed (numpy mrope is not jit-able) and fed as data.

Env (CPU 8-device emulation = multi-host proxy):
  export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive
  export JAX_PLATFORMS=cpu
  export XLA_FLAGS="--xla_force_host_platform_device_count=8"
"""
from __future__ import annotations

import argparse
import functools
import time
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from ddrive_jax import sharding
from ddrive_jax.models.qwen2_5_text import Qwen25TextConfig, Qwen25TextModel, mrope_cos_sin
from ddrive_jax.diffusion.masks import hybrid_block_causal_mask_dense, to_attn_mask4d
from ddrive_jax.diffusion.sasd_loss import section_weighted_ce, causal_ce
from ddrive_jax.data.grain_pipeline import make_sasd_loader
from ddrive_jax.train import dist, checkpoint_mgr

IMAGE_TOK = 151655
SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")


# ============================================================================ config ======
@dataclass
class HarnessConfig:
    """Everything the harness needs to (re)build the same model + shardings + optim."""
    proxy: bool = True
    dtype: object = jnp.float32
    # model (proxy defaults; real path overrides via Qwen25TextConfig.fast_ddrive)
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 32
    n_layers: int = 2
    mlp_hidden_size: int = 256
    vocab_size: int = 151936          # FULL vocab (exercises replicated-embedding path)
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    mrope_section: tuple = (4, 6, 6)  # proxy: 2*sum == head_dim(32); real (16,24,24)/128
    # sharding
    n_fsdp: int = 8
    n_tp: int = 1
    # optimizer
    opt: str = "adamw"                # "adamw" | "adafactor"
    lr: float = 1e-3
    clip: float = 1.0
    warmup_steps: int = 4
    total_steps: int = 40
    # proxy image embeds
    proxy_image_scale: float = 0.02
    seed: int = 0

    def text_config(self) -> Qwen25TextConfig:
        if not self.proxy:
            return Qwen25TextConfig.fast_ddrive(self.dtype)
        return Qwen25TextConfig(
            dtype=self.dtype, d_model=self.d_model, n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads, head_dim=self.head_dim, n_layers=self.n_layers,
            mlp_hidden_size=self.mlp_hidden_size, vocab_size=self.vocab_size,
            rms_norm_eps=self.rms_norm_eps, rope_theta=self.rope_theta,
            include_qkv_bias=True, include_bias=False, use_q_k_norm=False,
            tie_embeddings=True,
        )


# ======================================================================= optimizer ========
def build_tx(cfg: HarnessConfig):
    if cfg.warmup_steps > 0 and cfg.total_steps > cfg.warmup_steps:
        sched = optax.warmup_cosine_decay_schedule(
            0.0, cfg.lr, cfg.warmup_steps, cfg.total_steps - cfg.warmup_steps, cfg.lr * 0.1)
    else:
        sched = cfg.lr
    if cfg.opt == "adafactor":
        core = optax.adafactor(learning_rate=sched, multiply_by_parameter_scale=True,
                               min_dim_size_to_factor=128)
    else:
        core = optax.adamw(learning_rate=sched)
    return optax.chain(optax.clip_by_global_norm(cfg.clip), core)


# ===================================================================== host precompute ====
def prepare_batch(batch: dict, cfg: HarnessConfig, image_embeds_fn):
    """Host-side per-batch precompute: doubled cos/sin, mask4d, img_pos, image_embeds.

    Returns a dict of numpy/jax arrays ready to be device_put onto the data sharding tree
    and fed to the jitted train_step. `image_embeds_fn(B, twoN, D, step)` returns the
    frozen image embeds (proxy: stop_gradient(random); real: precomputed ViT, doubled).
    """
    tcfg = cfg.text_config()
    head_dim = tcfg.head_dim
    mrope_section = cfg.mrope_section
    B = batch["input_final"].shape[0]
    L = batch["rbi"].shape[1]

    cos_list, sin_list, mask_list, imgpos_list = [], [], [], []
    for b in range(B):
        pos2 = np.concatenate([batch["position_ids"][b], batch["position_ids"][b]], axis=1)  # [3,2L]
        cos_b, sin_b = mrope_cos_sin(pos2, head_dim, tcfg.rope_theta, mrope_section)          # [2L,hd]
        mask_b = to_attn_mask4d(hybrid_block_causal_mask_dense(
            jnp.asarray(batch["rbi"][b]), jnp.asarray(batch["turn"][b]), L))                  # [1,1,2L,2L]
        imgpos_b = np.where(batch["input_final"][b, 0] == IMAGE_TOK)[0]                       # [2N]
        cos_list.append(np.asarray(cos_b)); sin_list.append(np.asarray(sin_b))
        mask_list.append(np.asarray(mask_b)); imgpos_list.append(imgpos_b)

    twoN = imgpos_list[0].shape[0]
    assert all(x.shape[0] == twoN for x in imgpos_list), \
        f"non-uniform image-token count per sample: {[x.shape[0] for x in imgpos_list]}"

    cos = jnp.asarray(np.stack(cos_list))                # [B, 2L, hd]
    sin = jnp.asarray(np.stack(sin_list))                # [B, 2L, hd]
    mask4d = jnp.asarray(np.stack(mask_list))            # [B, 1, 1, 2L, 2L]
    img_pos = jnp.asarray(np.stack(imgpos_list).astype(np.int32))  # [B, 2N]
    image_embeds = image_embeds_fn(batch, B, twoN, tcfg.d_model, int(batch.get("step", 0)))  # [B,2N,D]
    assert img_pos.shape[1] == image_embeds.shape[1], \
        f"img_pos 2N={img_pos.shape[1]} != image_embeds 2N={image_embeds.shape[1]}"

    return {
        "input_final": jnp.asarray(batch["input_final"]),       # [B,2,2L] i64
        "labels_final": jnp.asarray(batch["labels_final"]),     # [B,2,L]  i64
        "original_labels": jnp.asarray(batch["original_labels"]),  # [B,1,L] i64
        "weights": jnp.asarray(batch["weights"]),               # [B,2,L]  f32
        "cos": cos, "sin": sin, "mask4d": mask4d,
        "img_pos": img_pos, "image_embeds": image_embeds,
        "num_items": jnp.asarray(batch["num_items"]).astype(jnp.float32),  # [B]
    }


PROXY_IMAGE_SEED = 1234  # FIXED: frozen ViT embeds are independent of MODEL init seed, so the
                         # proxy stand-in must be too (else two harnesses w/ different seeds get
                         # different "frozen" embeds -> breaks ckpt-resume loss continuity).


def proxy_image_embeds_fn(cfg: HarnessConfig):
    """Frozen dummy image embeds: stop_gradient(random*scale). Pure function of `step` only
    (model-seed-independent), mimicking real frozen ViT output."""
    def fn(batch, B, twoN, D, step):   # batch unused by the proxy (frozen random embeds)
        key = jax.random.fold_in(jax.random.PRNGKey(PROXY_IMAGE_SEED), int(step))
        ie = jax.random.normal(key, (B, twoN, D), dtype=cfg.dtype) * cfg.proxy_image_scale
        return jax.lax.stop_gradient(ie)
    return fn


# ====================================================================== data sharding =====
def _spec_by_rank(name: str, mesh):
    """P('fsdp', None...) on the leading B axis for each batch array."""
    rank = {
        "input_final": 3, "labels_final": 3, "original_labels": 3, "weights": 3,
        "cos": 3, "sin": 3, "mask4d": 5, "img_pos": 2, "image_embeds": 3, "num_items": 1,
    }[name]
    return NamedSharding(mesh, P("fsdp", *([None] * (rank - 1))))


def data_sharding_tree(mesh) -> dict:
    keys = ["input_final", "labels_final", "original_labels", "weights",
            "cos", "sin", "mask4d", "img_pos", "image_embeds", "num_items"]
    return {k: _spec_by_rank(k, mesh) for k in keys}


# ===================================================================== per-sample loss =====
def per_sample_loss_sums(model, ifn, lfn, ol, w, cos, sin, mask4d, img_pos, image_embeds):
    """The reference single-sample SASD forward, returning RAW CE sums (num_items=1.0).

    Shapes (no batch axis; vmap adds it):
      ifn [2,2L] i64, lfn [2,L], ol [1,L], w [2,L], cos/sin [2L,hd],
      mask4d [1,1,2L,2L] bool, img_pos [2N] i32, image_embeds [2N,D]
    """
    Lh = ifn.shape[1] // 2  # = L

    def embed_row(r):
        e = model.embed_tokens(ifn[r])                      # [2L, D]
        return e.at[img_pos].set(image_embeds.astype(e.dtype))
    emb2 = jnp.stack([embed_row(0), embed_row(1)], 0)       # [2, 2L, D]
    hidden = model.hidden_forward_mrope_cs(emb2, cos, sin, mask4d, remat=True)  # [2,2L,D]
    # TP hook (n_tp>1): lift this forward out of vmap and constrain [B,2,2L,D] with
    # P('fsdp',None,None,'tp'). For pure-FSDP the data+param shardings fully determine layout.
    nl = model.attend(hidden[:, :Lh, :])                    # [2, L, V] noisy half (both rows)
    cl = model.attend(hidden[:1, Lh:, :])                   # [1, L, V] clean half (mdm row)
    sec_sum = section_weighted_ce(nl, lfn, w, num_items=1.0)   # raw weighted sum
    causal_sum = causal_ce(cl, ol, num_items=1.0)              # raw sum
    return sec_sum, causal_sum


_DATA_KEYS = ("input_final", "labels_final", "original_labels", "weights",
              "cos", "sin", "mask4d", "img_pos", "image_embeds", "num_items")
FSDP = "fsdp"


def make_train_step(graphdef, tx, params_sharding, opt_sharding, data_shardings, repl, mesh):
    """Build the jitted train_step closed over graphdef + tx + mesh (static).

    DP+FSDP via `shard_map` over the 'fsdp' axis (the JAX-0.10 explicit-sharding-safe idiom;
    `vmap` over a mesh-sharded axis is rejected by sharding-in-types). Inside the shard:
      - params enter REPLICATED (resharded P() at the boundary -> FSDP forward all-gather),
      - each device sees its local batch shard [B/n_fsdp, ...],
      - it computes raw section/causal CE sums + local grads over its shard,
      - `psum('fsdp')` reduces numerator / denominator / grads across devices (DP-correct).
    The reduced grads come out replicated; we reshard them to params_sharding and apply optax.
    """
    data_in_specs = {k: data_shardings[k].spec for k in _DATA_KEYS}

    def local_grads(params, batch):
        """Runs on ONE device's local batch shard; returns psum'd global (loss, aux, grads)."""
        def raw_sums(params):
            model = nnx.merge(graphdef, params)
            sec_sum, causal_sum = jax.vmap(
                per_sample_loss_sums, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
                model,
                batch["input_final"], batch["labels_final"], batch["original_labels"],
                batch["weights"], batch["cos"], batch["sin"], batch["mask4d"],
                batch["img_pos"], batch["image_embeds"])
            return jnp.sum(sec_sum) + jnp.sum(causal_sum), (jnp.sum(sec_sum), jnp.sum(causal_sum))
        (local_num, (sec, cau)), grads = jax.value_and_grad(raw_sums, has_aux=True)(params)
        local_den = jnp.sum(batch["num_items"])
        # reduce across the data-parallel ('fsdp') shards -> GLOBAL sums (DP-correct)
        g_num = jax.lax.psum(local_num, FSDP)
        g_den = jax.lax.psum(local_den, FSDP)
        g_sec = jax.lax.psum(sec, FSDP)
        g_cau = jax.lax.psum(cau, FSDP)
        grads = jax.lax.psum(grads, FSDP)
        loss = g_num / jnp.maximum(g_den, 1.0)
        return loss, (g_sec, g_cau, g_den), grads

    @functools.partial(
        jax.jit,
        in_shardings=(params_sharding, opt_sharding, data_shardings),
        out_shardings=(params_sharding, opt_sharding, repl, repl),
        donate_argnums=(0, 1),
    )
    def train_step(params, opt_state, batch):
        params_full = jax.reshard(params, jax.tree.map(lambda _: P(), params))
        f = jax.shard_map(
            local_grads, mesh=mesh,
            in_specs=(jax.tree.map(lambda _: P(), params), data_in_specs),
            out_specs=(P(), (P(), P(), P()), jax.tree.map(lambda _: P(), params)))
        loss, aux, grads = f(params_full, batch)
        grads = jax.reshard(grads, params_sharding)   # grads come out replicated -> reshard
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    return train_step


# ============================================================================= harness ====
@dataclass
class Harness:
    cfg: HarnessConfig
    mesh: object
    model: object
    graphdef: object
    tx: object
    params: object
    opt_state: object
    params_sharding: object
    opt_sharding: object
    data_shardings: dict
    repl: object
    train_step: object
    image_embeds_fn: object
    abstract_params: object = field(default=None)
    abstract_opt: object = field(default=None)


def _opt_pspec_for_leaf(leaf, embed_shape):
    """Same fsdp rule as params; embedding-shaped leaf forced replicated (count -> P())."""
    shape = tuple(leaf.shape)
    if shape == embed_shape:
        return P()
    return sharding.fsdp_pspec(shape)


def build_harness(cfg: HarnessConfig) -> Harness:
    """Construct the full sharded harness (model, params, opt, shardings, jitted step)."""
    mesh = dist.build_mesh(cfg.n_fsdp, cfg.n_tp)
    tcfg = cfg.text_config()
    embed_shape = (tcfg.vocab_size, tcfg.d_model)

    with dist.mesh_context(mesh):
        if cfg.proxy:
            model = Qwen25TextModel(tcfg, rngs=nnx.Rngs(cfg.seed))
        else:
            from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_text
            model, _ = load_fast_ddrive_text(SNAP, tcfg, dtype=cfg.dtype, verbose=False)  # pretrained
        graphdef = nnx.graphdef(model)

        # param shardings (leaf-aligned with nnx.state(model, nnx.Param))
        pspecs = sharding.param_pspecs(model, replicate_embedding=True)
        params_sharding = jax.tree.map(
            lambda ps: NamedSharding(mesh, ps), pspecs, is_leaf=lambda x: isinstance(x, P))
        params = nnx.state(model, nnx.Param)
        params = jax.device_put(params, params_sharding)

        # optimizer state, initialised already sharded
        tx = build_tx(cfg)
        abstract_opt = jax.eval_shape(lambda p: tx.init(p), params)
        opt_pspecs = jax.tree.map(lambda l: _opt_pspec_for_leaf(l, embed_shape), abstract_opt)
        opt_sharding = jax.tree.map(
            lambda ps: NamedSharding(mesh, ps), opt_pspecs, is_leaf=lambda x: isinstance(x, P))
        opt_state = jax.jit(lambda p: tx.init(p), out_shardings=opt_sharding)(params)

        repl = NamedSharding(mesh, P())
        data_shardings = data_sharding_tree(mesh)
        train_step = make_train_step(graphdef, tx, params_sharding, opt_sharding,
                                     data_shardings, repl, mesh)

    abstract_params = jax.eval_shape(lambda: nnx.state(model, nnx.Param))
    image_embeds_fn = proxy_image_embeds_fn(cfg) if cfg.proxy else _real_image_embeds_fn(cfg)

    return Harness(
        cfg=cfg, mesh=mesh, model=model, graphdef=graphdef, tx=tx,
        params=params, opt_state=opt_state,
        params_sharding=params_sharding, opt_sharding=opt_sharding,
        data_shardings=data_shardings, repl=repl, train_step=train_step,
        image_embeds_fn=image_embeds_fn,
        abstract_params=abstract_params, abstract_opt=abstract_opt,
    )


def _real_image_embeds_fn(cfg: HarnessConfig):
    """Real frozen ViT image embeds (GPU/TPU). Loads the Fast-dDrive ViT once; per batch runs it
    on each sample's pixel_values and doubles to [2N, D] (matching the doubled [noisy|clean]
    sequence), then stacks to [B, 2N, D]. Mirrors train_waymo_sasd_jax.static_tensors. Frozen
    (stop_gradient), so ViT params never receive gradients."""
    from ddrive_jax.models.vision_qwen25vl import VisionConfig
    from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_vit
    vit, _ = load_fast_ddrive_vit(SNAP, VisionConfig(dtype=cfg.dtype), dtype=cfg.dtype, verbose=False)

    def fn(batch, B, twoN, D, step):
        outs = []
        for b in range(B):
            ie = vit(jnp.asarray(batch["pixel_values"][b], cfg.dtype),
                     np.asarray(batch["image_grid_thw"][b]))     # [N, D]
            ie = jnp.concatenate([ie, ie], 0).astype(cfg.dtype)  # [2N, D] (doubled sequence)
            outs.append(jax.lax.stop_gradient(ie))
        return jnp.stack(outs, 0)                                # [B, 2N, D]
    return fn


def _shard_inputs(jit_inputs: dict, data_shardings: dict):
    """Place each host-local batch leaf onto its (global) data sharding.

    Single process (local GPU / CPU-N emulation, process_count==1): ``device_put`` is exact and
    keeps the verified path byte-identical. True multi-host (>=2 processes, e.g. a 2-VM TPU
    slice): each process only addresses its OWN local devices, so ``device_put`` of a host-LOCAL
    array onto a GLOBAL sharding is wrong -- it would treat this process's shard as the whole
    array (and raise, since the shard can't cover non-addressable devices). The grain loader
    already yields per-host shards (process_index/count), so the per-step global batch is
    ``per_host_batch * process_count`` distinct samples; assemble the global ``jax.Array`` from
    the per-process-local data. Sample->device placement is irrelevant to the DP loss (a psum
    over ALL shards), so any consistent assembly is correct.
    """
    if jax.process_count() == 1:
        return jax.device_put(jit_inputs, data_shardings)
    pc = jax.process_count()

    def to_global(local, sharding):
        local = np.asarray(local)
        global_shape = (local.shape[0] * pc,) + local.shape[1:]
        return jax.make_array_from_process_local_data(sharding, local, global_shape)

    return {k: to_global(jit_inputs[k], data_shardings[k]) for k in jit_inputs}


def run_step(h: Harness, batch: dict):
    """Host-prepare a grain batch + run one sharded train_step. Returns (loss, aux)."""
    jit_inputs = prepare_batch(batch, h.cfg, h.image_embeds_fn)
    with dist.mesh_context(h.mesh):
        jit_inputs = _shard_inputs(jit_inputs, h.data_shardings)
        h.params, h.opt_state, loss, aux = h.train_step(h.params, h.opt_state, jit_inputs)
    return loss, aux


# ============================================================================== driver =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", action="store_true", help="small proxy model (CPU verify)")
    ap.add_argument("--parquet_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch", type=int, default=8, help="per-host batch")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--ckpt_dir", default="/tmp/ddrive_proxy_ckpt")
    ap.add_argument("--opt", default="adamw", choices=["adamw", "adafactor"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--n_fsdp", type=int, default=0, help="0 -> jax.device_count()")
    ap.add_argument("--n_tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    dist.init_distributed()
    n_fsdp = args.n_fsdp or jax.device_count()
    dtype = jnp.bfloat16 if args.bf16 else jnp.float32

    cfg = HarnessConfig(
        proxy=args.proxy, dtype=dtype, n_fsdp=n_fsdp, n_tp=args.n_tp,
        opt=args.opt, lr=args.lr, clip=args.clip,
        warmup_steps=max(args.steps // 10, 1), total_steps=args.steps, seed=args.seed,
    )
    if not args.proxy:
        # real model dims (FULL config); proxy mrope_section default is wrong for hd=128.
        cfg.mrope_section = (16, 24, 24)

    dist.pprint(f"[harness] proxy={cfg.proxy} mesh=({n_fsdp},{args.n_tp}) "
                f"opt={cfg.opt} lr={cfg.lr} dtype={dtype.__name__} "
                f"devices={jax.device_count()} procs={jax.process_count()}")

    h = build_harness(cfg)
    dist.pprint(f"[harness] built; params sharded, opt sharded.")

    loader = make_sasd_loader(
        args.parquet_dir, args.split, per_host_batch=args.batch, seed=args.seed,
        process_index=jax.process_index(), process_count=jax.process_count())
    it = iter(loader)

    mgr = checkpoint_mgr.build_manager(
        args.ckpt_dir, save_interval_steps=args.save_every, max_to_keep=args.keep)
    start_step = 1
    restored = checkpoint_mgr.restore_latest(
        mgr, h.abstract_params, h.abstract_opt, h.params_sharding, h.opt_sharding)
    if restored is not None:
        step, grain_state, rparams, ropt = restored
        h.params, h.opt_state = rparams, ropt
        loader.set_state({"grain": grain_state["grain"]})
        start_step = step + 1
        dist.pprint(f"[harness] restored step={step}; resuming at {start_step}")

    t0 = time.time()
    losses = []
    for st in range(start_step, args.steps + 1):
        batch = next(it)
        loss, aux = run_step(h, batch)
        lv = float(loss)
        losses.append(lv)
        if st % 5 == 0 or st == start_step:
            dist.pprint(f"step {st:4d} loss {lv:.4f} [{(time.time()-t0)/max(st-start_step+1,1)*1000:.0f} ms/step]")
        checkpoint_mgr.save_step(mgr, st, h.params, h.opt_state, loader.state(),
                                 extra_meta={"mrope_section": list(cfg.mrope_section),
                                             "n_fsdp": n_fsdp})
    checkpoint_mgr.wait(mgr)
    dist.pprint(f"[done] {len(losses)} steps; loss {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
