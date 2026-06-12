"""Generate a self-contained static website explaining the SASD **v2** training dataset:
what a processed record looks like (incl. precomputed `image_embeds`), how it is built
from raw WOD-E2E, and how it is fed into the model — illustrated with the 12 verified
round-2 examples (10 train + 2 val).

Reads the round-2 v2 artifacts under /home/kaiwen/data/fast-ddrive/verify_round2_v2/
(ar_npz, embeds_check, viz_data.json, review PNGs, src.json) and writes a static site
(pure relative paths, no CDN, no JS dependencies) to jax_ddrive/visualizations/.

Run (ddrive env):
  PYTHONPATH=jax_ddrive /home/kaiwen/miniconda3/envs/ddrive/bin/python \
      jax_ddrive/scripts/make_dataset_website.py
"""
import glob
import html
import json
import os
import re
import shutil
import sys

import numpy as np

REPO = "/home/kaiwen/Desktop/research/Fast-dLLM"
SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
VR = "/home/kaiwen/data/fast-ddrive/verify_round2_v2"
HF = "/home/kaiwen/data/fast-ddrive/hf"
OUT = REPO + "/jax_ddrive/visualizations"

sys.path.insert(0, REPO + "/jax_ddrive")
sys.path.insert(0, REPO + "/jax_ddrive/scripts")
from verify_ar_round2 import (_render_collapsed, _runs, IMAGE_PAD, MASK_ID,  # noqa: E402
                              IM_START, IM_END)
from ddrive_jax.diffusion import noise as noise_mod                          # noqa: E402

E = html.escape

CSS = """
:root { --bg:#f7f8fa; --card:#fff; --ink:#1c2733; --mut:#6b7a8c; --acc:#0b66c3;
        --ok:#1d8a4e; --line:#e3e8ee; --code:#f1f4f8; --v2:#7c3aed; }
* { box-sizing: border-box; }
body { margin:0; font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
       "Helvetica Neue",Arial,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
       color:var(--ink); background:var(--bg); }
nav { position:sticky; top:0; z-index:9; background:#102a43; color:#d9e2ec;
      padding:10px 26px; display:flex; gap:18px; align-items:center; flex-wrap:wrap; }
nav a { color:#d9e2ec; text-decoration:none; font-weight:600; font-size:14px; }
nav a:hover { color:#fff; text-decoration:underline; }
nav .brand { color:#fff; font-size:16px; margin-right:8px; }
main { max-width:1180px; margin:0 auto; padding:28px 26px 80px; }
h1 { font-size:26px; margin:18px 0 6px; }
h2 { font-size:20px; margin:38px 0 10px; padding-top:8px; border-top:2px solid var(--line); }
h3 { font-size:16px; margin:22px 0 8px; }
.sub { color:var(--mut); margin:0 0 18px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:16px 20px; margin:14px 0; }
.grid { display:grid; gap:14px; }
.grid.c2 { grid-template-columns:1fr 1fr; } .grid.c3 { grid-template-columns:1fr 1fr 1fr; }
@media (max-width:900px){ .grid.c2,.grid.c3{grid-template-columns:1fr;} }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th,td { border:1px solid var(--line); padding:6px 10px; text-align:left; vertical-align:top; }
th { background:#eef2f7; }
code, .tok { font:12.5px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
       background:var(--code); padding:1px 5px; border-radius:4px; }
pre { font:12.5px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
      background:var(--code); border:1px solid var(--line); border-radius:8px;
      padding:12px 14px; overflow-x:auto; white-space:pre-wrap; word-break:break-word; }
img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:8px;
      background:#fff; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12.5px;
         font-weight:700; }
.badge.ok { background:#e3f6ec; color:var(--ok); }
.badge.info { background:#e4eefb; color:var(--acc); }
.badge.v2 { background:#f1e9fe; color:var(--v2); }
.badge.val { background:#fdebd0; color:#9c640c; }
details { margin:10px 0; }
details > summary { cursor:pointer; font-weight:600; color:var(--acc); }
.flow { display:flex; flex-direction:column; gap:0; margin:18px 0; }
.stage { background:var(--card); border:1px solid var(--line); border-left:5px solid var(--acc);
         border-radius:10px; padding:12px 18px; }
.stage.v2 { border-left-color:var(--v2); }
.stage h3 { margin:2px 0 6px; }
.stage .env { float:right; font-size:12px; color:var(--mut); }
.arrow { text-align:center; color:var(--mut); font-size:20px; line-height:1.4; }
.kv { color:var(--mut); font-size:13.5px; }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:16px; }
.scard { background:var(--card); border:1px solid var(--line); border-radius:10px;
         padding:12px; text-decoration:none; color:var(--ink); transition:box-shadow .15s; }
.scard:hover { box-shadow:0 4px 14px rgba(16,42,67,.12); }
.scard img { width:100%; }
.scard .t { font:12px ui-monospace,Menlo,monospace; color:var(--mut); margin-top:6px;
            word-break:break-all; }
.pn { display:flex; justify-content:space-between; margin:26px 0 8px; }
.pn a { color:var(--acc); font-weight:700; text-decoration:none; }
.note { background:#fff8e6; border:1px solid #f1de9d; border-radius:8px; padding:10px 14px;
        font-size:13.5px; }
"""


def nav(n_samples):
    links = "".join(f'<a href="sample_{k:02d}.html">{k:02d}</a>' for k in range(n_samples))
    return (f'<nav><span class="brand">Fast-dDrive SASD dataset v2 review</span>'
            f'<a href="index.html">总览 / Pipeline / 喂给模型</a>'
            f'<span style="color:#829ab1">样本:</span>{links}</nav>')


def page(title, body, n_samples):
    return (f'<!doctype html><html lang="zh"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{E(title)}</title><style>{CSS}</style></head>'
            f'<body>{nav(n_samples)}<main>{body}</main></body></html>')


def parse_checks(review_md):
    txt = open(review_md).read()
    sec = txt.split("## checks", 1)[1].split("##", 1)[0]
    return re.findall(r"^\| (\S+) \| (✅|❌) \|$", sec, re.M)


def fmt_bytes(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


SAMPLE_PNGS = ["triptych__orig_recon_embedsPCA.png", "side_by_side__orig_vs_recon.png",
               "embeds_recompute_diff.png", "bev_trajectory.png",
               "sequence_layout.png", "position_ids.png", "attn_mask.png"]


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)

    loc = json.load(open(os.path.join(VR, "locations.json")))
    src = {s["sample_id"]: s for s in json.load(open(os.path.join(VR, "src.json")))}
    os.makedirs(OUT, exist_ok=True)
    # clear stale per-sample pages/assets from previous (v1) generations
    for p in glob.glob(os.path.join(OUT, "sample_*.html")):
        os.remove(p)
    shutil.rmtree(os.path.join(OUT, "assets"), ignore_errors=True)

    sizes = {d: sum(os.path.getsize(f) for f in glob.glob(os.path.join(HF, d, "*")))
             for d in ["wod_e2e_sasd_full_v2_ar", "wod_e2e_sasd_50k_v2_ar",
                       "wod_e2e_sasd_val_v2_ar", "wod_e2e_sasd_v2_ar"]}

    samples = []
    for k, sid in enumerate(loc["order"]):
        split = loc["splits"][sid]
        sdir = os.path.join(VR, "review", f"{k:02d}_{sid}")
        adir = os.path.join(OUT, "assets", f"{k:02d}")
        os.makedirs(adir, exist_ok=True)
        for png in SAMPLE_PNGS:
            shutil.copy2(os.path.join(sdir, png), os.path.join(adir, png))
        item = src[sid]
        for i, rel in enumerate(item["image"]):
            shutil.copy2(os.path.join(VR, "src_images", rel),
                         os.path.join(adir, f"orig_{i}.jpg"))

        ar = dict(np.load(os.path.join(VR, "ar_npz", sid + ".npz")))
        ids, labels, rbi = ar["input_ids"], ar["labels"], ar["rbi"]
        L = int(ids.shape[0])
        resp = np.where(labels != -100)[0]
        rs, re_ = int(resp[0]), int(resp[-1]) + 1
        vd = json.load(open(os.path.join(sdir, "viz_data.json")))
        checks = parse_checks(os.path.join(sdir, "review.md"))
        pad_runs = [(s, ln) for t, s, ln in _runs(list(ids)) if t == IMAGE_PAD]

        ans_decoded = tok.decode(ids[rs:re_ - 1], skip_special_tokens=False)
        full_decoded = _render_collapsed(ids, tok)
        ifn, lfn, _ol, _w = noise_mod.make_batch(
            {kk: ar[kk] for kk in ("input_ids", "labels", "rbi", "scaffold",
                                   "weight_vec", "block_alpha", "block_beta")},
            np.random.default_rng(0))
        noisy_answer = _render_collapsed(ifn[0, max(rs - 4, 0):L], tok)
        comp_answer = _render_collapsed(ifn[1, max(rs - 4, 0):L], tok)
        n_masked = int((ifn[0, :L] == MASK_ID).sum())
        n_comp = int((ifn[1, :L] == MASK_ID).sum())

        b2s = {int(b): s for b, s in vd["block_to_section"].items()}
        sec_rows = []
        for b in sorted(b2s):
            pos = int(np.sum(rbi == b))
            w = float(np.max(ar["weight_vec"][rbi == b])) if pos else 0
            sec_rows.append((b, b2s[b], pos, w,
                             float(ar["block_alpha"][b]), float(ar["block_beta"][b])))

        samples.append(dict(
            k=k, sid=sid, split=split, L=L, rs=rs, re=re_, vd=vd, checks=checks,
            n_blocks=int(ar["n_blocks"]), grid=ar["image_grid_thw"].tolist(),
            runs=[ln for _, ln in pad_runs], npix=int(ar["pixel_values"].shape[0]),
            mask_tail=int((ids[re_:] == MASK_ID).sum()),
            prompt=item["conversations"][0]["value"],
            gpt_raw=item["conversations"][1]["value"],
            ans=ans_decoded, full=full_decoded,
            noisy=noisy_answer, comp=comp_answer, n_masked=n_masked, n_comp=n_comp,
            n_resp=int((labels != -100).sum()),
            loc=loc["locations"][sid], sec_rows=sec_rows,
            emb=vd["embeds"],
        ))

    write_index(samples, sizes)
    for s in samples:
        write_sample(s, len(samples))
    write_readme()
    print(f"[website] index + {len(samples)} sample pages -> {OUT}")


# ───────────────────────────────────────────────────────── index sections ────
def write_index(S, sizes):
    n = len(S)
    s0 = S[0]
    schema_rows = [
        ("sample_id", "str", "WOD-E2E frame id（frame.context.name），可 join 回 GT", "追溯/调试"),
        ("input_ids", "[1184] int64", "完整 chat-template token 序列（结构见下文布局图）", "embed 后做模型输入"),
        ("labels", "[1184] int64", "answer 区间 = token id；其余 -100（不计 loss）", "loss 目标"),
        ("rbi", "[1184] int32", "response block index：prompt=-1；answer 按 section 切成 ≤32 value-token 的块", "构建注意力 mask + 分配 section"),
        ("turn", "[1184] int32", "块边界处 +1 的递增序号", "训练 mask 的 turn 规则"),
        ("scaffold", "[1184] bool", "JSON 骨架位置（key、引号、分隔符）——永不加噪、永不训练", "加噪时冻结"),
        ("weight_vec", "[1184] f32", "逐 token loss 权重：CO 1.5 / explanation 1.0 / fmb 2.0 / trajectory 3.0", "section-weighted CE"),
        ("block_alpha / block_beta", f"[{s0['n_blocks']}] f32", "每块 Beta(α,β) 噪声调度：CO(1,2) exp(1,1) fmb(1,1.5) traj(2,1)", "在线加噪采样 mask 比例"),
        ("position_ids", "[3,1184] int32", "3D M-RoPE 位置（temporal/height/width 三通道）", "RoPE cos/sin"),
        ("vision_mask", "[1184] bool", "vision token 位置（image_pad / vision_start / vision_pad）", "辅助/调试"),
        ("pixel_values", f"[{s0['npix']},1176] f16", "3 张图的像素 patch：每行=14×14×3 通道×2 时间帧，CLIP 归一化（可逆；保留它使 embeds 永远可独立复验）", "复验源（v2 训练不再喂 ViT）"),
        ("image_grid_thw", "[3,3] int64", f"每图 (t,h,w) patch 网格 = {s0['grid'][0]}", "索引 + M-RoPE"),
        ("<b>image_embeds</b> ⭐v2", "[168,2048] bf16", "frozen ViT 输出<b>离线预计算</b>(fp32 highest → bf16,单份;loader `concat([ie,ie])` 翻倍到 336)——训练时直接散射到占位符位置,pod 上不再跑 ViT", "<b>模型的图像输入</b>"),
        ("L / n_blocks", "标量", "序列长 / response 块数", "静态形状"),
    ]
    schema_html = "".join(
        f"<tr><td><code>{a}</code></td><td><code>{E(b)}</code></td><td>{c}</td><td>{d}</td></tr>"
        for a, b, c, d in schema_rows)

    stages = [
        ("Stage 0 — 原始数据", "—", "",
         "WOD-E2E tfrecords(263 train + 93 val shards,~1.1 TB)。每帧一个 <code>E2EDFrame</code> proto:"
         "8 路相机 JPEG、16 点自车历史(@0.25s)、20 点未来轨迹(@4Hz,<b>真值</b>)、导航 intent、rater 偏好轨迹。",
         "<code>/home/kaiwen/data/fast-ddrive/waymo/</code>"),
        ("Stage 1 — 抽取 + 造 prompt/target", "autovla env",
         "✅ prompt 对官方 sample.json 逐字节验证",
         "<code>fast_ddrive/data/convert_wod_e2e.py --with_target</code>:取<b>左前/正前/右前</b> 3 路相机存 JPEG;"
         "固定指令模板 + 导航命令 + 7 点自车历史(@0.5s)拼 prompt;target JSON 里 <b>trajectory=真值 5 点@1s</b>,"
         "future_meta_behavior 派生,critical_objects/explanation 是伪标签(已知限制,可 teacher-distill 升级)。"
         "val split 加 <code>--rated_only</code>(479 个 rater 评分帧)。",
         "输出:train/val JSON + JPEGs"),
        ("Stage 2 — Token 化 + SASD 结构", "ddrive env",
         "✅ round-2:12 列 vs 从 raw 重跑 bit-exact",
         "<code>jax_ddrive/eval/prep_train_jax.py</code>:HF processor(chat template;每图 16×14 patch 网格);"
         "<code>process_gpt</code> 规范化 answer;标 labels;deep-scaffold 检测 scaffold/value、按 section 切块"
         "(rbi/turn)、配权重和 Beta(α,β);3D M-RoPE position_ids;pad 到 L=1184。",
         "输出:每帧一个 npz(12 个数组)"),
        ("Stage 3 — Parquet(bit-exact 真相源)", "pyarrow",
         "✅ 三方计数对账;packed=唯一本地 parquet 源",
         "<code>prep_to_parquet.py</code>(原始字节 blob + shape 列,无损)→ "
         "<code>pack_parquet.py</code>(纯 concat 合并成 130 个文件,内容不变)。",
         "输出:<code>wod_e2e_sasd_full_packed/</code> 178G"),
        ("Stage 4 — AR v2:预计算 ViT embeds ⭐", "jax venv + 5090 GPU",
         "✅ 构建期四道内建检查:每 shard jit-vs-eager ViT 抽查(观测 ≤5e-6,硬上限 1e-3)· bitwise recompute + bf16 roundtrip · 计数对账 · AR↔parquet 逐字节(test_ar_pipeline.py)",
         "<code>jax_ddrive/scripts/parquet_to_ar_with_embeds.py</code>(~19 样本/s,原子可断点续):"
         "对每帧跑 frozen ViT(fp32 highest)→ bf16 存为 <code>image_embeds</code>,连同 13 个原字段一起打成 "
         "ArrayRecord(tf.train.Example)。<b>这就是 TPU 训练直接消费的生产格式。</b>",
         "输出:v2 AR 四个 split(下表)"),
    ]
    stages_html = "".join(
        f'<div class="stage{" v2" if "⭐" in name else ""}"><span class="env">{env}</span>'
        f'<h3>{name}</h3>'
        + (f'<p class="kv" style="color:var(--ok)">{ver}</p>' if ver else "")
        + f'<p>{desc}</p><p class="kv">{io}</p></div>'
        + ('<div class="arrow">⬇︎</div>' if i < len(stages) - 1 else "")
        for i, (name, env, ver, desc, io) in enumerate(stages))

    split_rows = [
        ("wod_e2e_sasd_full_v2_ar", "415,663", "130", "生产训练(full)"),
        ("wod_e2e_sasd_50k_v2_ar", "50,331", "787", "shakeout 训练"),
        ("wod_e2e_sasd_val_v2_ar", "479", "8", "val(rated 帧,训练格式带 target)"),
        ("wod_e2e_sasd_v2_ar", "400", "7", "烟测/单测"),
    ]
    splits_html = "".join(
        f"<tr><td><code>{d}</code></td><td>{r}</td><td>{sh}</td>"
        f"<td>{fmt_bytes(sizes[d])}</td><td>{role}</td></tr>"
        for d, r, sh, role in split_rows)

    cards = "".join(
        f'<a class="scard" href="sample_{s["k"]:02d}.html">'
        f'<img src="assets/{s["k"]:02d}/triptych__orig_recon_embedsPCA.png" loading="lazy" '
        f'alt="sample {s["k"]:02d}">'
        f'<div><span class="badge ok">✅ {sum(1 for _, v in s["checks"] if v == "✅")}/'
        f'{len(s["checks"])} checks</span> '
        + (f'<span class="badge val">val</span> ' if s["split"] == "val"
           else '<span class="badge info">train</span> ')
        + f'<span class="badge v2">emb {s["emb"]["bitwise_frac"]:.2%}</span></div>'
        f'<div class="t">{s["k"]:02d} · {E(s["sid"])}</div></a>'
        for s in S)

    n_train = sum(1 for s in S if s["split"] == "train")
    n_val = n - n_train
    body = f"""
<h1>Fast-dDrive SASD 训练数据集 <span class="badge v2">v2</span> — 是什么、怎么来的、怎么喂给模型</h1>
<p class="sub">生产格式:<code>wod_e2e_sasd_*_v2_ar</code>(ArrayRecord,13 原字段 + 预计算
<code>image_embeds</code>)· 全量 415,663 帧 · 本站 {n} 个样本({n_train} train + {n_val} val)
均通过 round-2 验证:12 列 vs 从 raw 重跑 bit-exact + embeds 从像素复算两层判据</p>

<div class="card"><b>一句话:</b> 每条记录 = 一个驾驶瞬间。<b>三张前视相机图</b>(像素在
<code>pixel_values</code>,<b>v2 还存了它们的 frozen-ViT 输出 <code>image_embeds</code></b> ——
训练时直接用,pod 上不再跑 ViT)+ <b>文本 prompt</b>(任务指令、导航命令、3 秒自车历史)
+ <b>JSON 答案</b>(critical_objects / explanation / future_meta_behavior / <b>真值轨迹</b>),
外加一套 SASD 训练结构(哪些 token 可加噪、各属哪个 section、权重和噪声调度)。</div>

<div class="card"><h3 style="margin-top:0">v1 → v2 变了什么</h3>
<table><tr><th></th><th>AR v1(已退役,仅存 GCS)</th><th>AR v2(生产格式)</th></tr>
<tr><td>字段</td><td>12 数组 + 3 标量(只有像素)</td><td>同样 13 个字段全保留 + <b>image_embeds bf16 [168,2048]</b></td></tr>
<tr><td>训练时图像处理</td><td>每 step 在 host 跑 frozen ViT(数据循环被卡住)</td><td>读现成 embeds → 纯 IO;pod 零 ViT/transformers 依赖</td></tr>
<tr><td>v6e-1 实测吞吐</td><td>65 TFLOP/s/device</td><td><b>83.4 TFLOP/s/device(+28%)</b></td></tr>
<tr><td>谁能读</td><td>无训练路径读它(都走 parquet)</td><td><code>make_sasd_loader</code> 按扩展名自动探测;FSDP harness 与 MaxText 直接消费;iterator 状态进 ckpt → <b>resume 续流不重放</b>(v6e-1 实测续训 12..17)</td></tr>
<tr><td>split</td><td>train only</td><td>full / 50k / 400 / <b>val 479(训练格式带 target)</b></td></tr></table></div>

<h2 id="record">① 处理后的一条记录长什么样</h2>
<p>14 个字段。注意:<code>input_ids</code> 中的"图像部分"只是 56×3 个相同的占位符
<code>&lt;|image_pad|&gt;</code>(id 151655);像素在 <code>pixel_values</code>,
<b>模型实际吃的图像表示在 <code>image_embeds</code></b>(forward 时散射到占位符位置)。</p>
<table><tr><th>字段</th><th>形状/类型</th><th>含义</th><th>训练时用途</th></tr>{schema_html}</table>

<h3>一个样本的序列布局(样本 00,L=1184 个 token 各是什么)</h3>
<img src="assets/00/sequence_layout.png" alt="sequence layout">
<p class="kv">上条:区域/section(灰=prompt 文本,蓝=图像占位区,橙/绿/紫/红=答案四个 section,黑=MASK pad)。
下条:训练角色——深灰 scaffold(JSON 骨架,冻结)、红 value(被加噪+训练的内容)、黑 pad。</p>

<h2 id="pipeline">② 从 raw 数据怎么处理出来的(每级带验证)</h2>
<div class="flow">{stages_html}</div>
<h3>v2 AR 四个 split(本地 + GCS 镜像)</h3>
<table><tr><th>目录(/home/kaiwen/data/fast-ddrive/hf/)</th><th>行数</th><th>shards</th><th>大小</th><th>角色</th></tr>
{splits_html}</table>
<div class="note">本站验证(round-2,2026-06-12):对 {n} 个样本从 raw tfrecord 重跑整条链,
12 个数组列与 v2 AR 记录<b>逐列 bit-exact</b>;<code>image_embeds</code> 用存储的像素重新过
frozen ViT(fp32 highest → bf16)比对,两层判据 = bitwise ≥98.5%(实测 ~99.8%)且每个
mismatch ≤1.5× 自身 bf16 ulp 或 |Δ|≤2e-3(全部通过;bf16 级全局 rel 仅作诊断报告——单个大幅值
元素的 1-ulp 翻转就有 ~2⁻⁸ 相对量级,对它设门限是范畴错误)。端到端旁证:同一 prep 管线的
479 帧 val 上 JAX/PyTorch 官方指标 ADE@3s 0.839/0.814、RFS 7.93/7.91 持平。</div>

<h2 id="feed">③ 训练时怎么喂给模型(每个 train step,v2 路径)</h2>
<div class="card">
<p><b>1. 读取:</b> <code>make_sasd_loader</code> 按扩展名自动探测 AR(lazy 随机访问,全量可用)/
parquet;按 host 切分全局 shuffle 流,确定性、可断点续;<b>iterator 状态随 checkpoint 保存,
resume 续流不重放</b>。batch 里直接带 <code>image_embeds (B,168,2048) bf16</code>。</p>
<p><b>2. 在线加噪(<code>noise.make_batch</code>):</b> 每个 response 块抽
<code>t ~ Beta(α,β)</code> → 以概率 <code>p=(1-ε)t+ε</code> 把该块的 <b>value</b> token 替换成
<code>|&lt;MASK&gt;|</code>(scaffold 冻结;<code>&lt;|im_end|&gt;</code> 必 mask)。轨迹块 α,β=(2,1)
→ 平均被 mask 更多(任务更难、权重也最高 3.0)。</p>
<p><b>3. 拼接 doubled 序列(2L=2368),每样本两行:</b></p>
<pre>row 0 (mdm):  [ x_t  = 答案被随机 mask     | x_0 = 干净原序列 ]   ← labels: 只在被 mask 的位置
row 1 (comp): [ x̄_t = 互补位置被 mask      | x_0 = 干净原序列 ]   ← labels: 互补位置
</pre>
<p><b>4. 图像(v2):</b> 直接取记录里的 <code>image_embeds</code> →
<code>concat([ie,ie])</code> 翻倍到 [336,2048] → 散射到两行序列里 <code>&lt;|image_pad|&gt;</code>
的位置。<b>不再有任何 ViT 前向</b>(v1 在这里每 step 跑一次 frozen ViT —— 数据循环因此从
65 提到 83.4 TFLOP/s/device)。</p>
<p><b>5. 位置 & 注意力:</b> position_ids 平铺两份 → M-RoPE cos/sin(contiguous-chunk 布局,
非 MaxText 内建 interleaved);由 rbi/turn 构建 [2L,2L] hybrid block-causal mask(下图)。</p>
<p><b>6. loss:</b>
<code>Σ section_weighted_CE(noisy 半边 logits, labels, w) + causal_CE(clean 半边 row0, 原 labels)</code>,
除以 <code>2×答案 token 数</code>。消费方:NNX FSDP harness(<code>train_tpu.py</code>)和
MaxText <code>objective="sasd"</code>(fork @ a645b25,vendored 自包含 <code>sasd_data/</code>)。</p>
</div>
<div class="grid c2">
<div><img src="assets/00/attn_mask.png" alt="attention mask">
<p class="kv">训练注意力 mask(样本 00):左上=noisy 半边(prompt 整块双向 + 答案逐块);
右上=noisy→clean 只看更早的 turn(阶梯);右下=clean 半边标准 causal。</p></div>
<div><img src="assets/00/position_ids.png" alt="position ids">
<p class="kv">3D M-RoPE:文本区三通道重合对角线;图像区 t 持平、h/w 锯齿;图像段只前进
max(h,w),所以末端位置 ≈1040 &lt; 1184。</p></div>
</div>
<p>每个样本页底部都有"加噪后模型实际看到的序列"文本示例(mdm 行 + 互补行)。</p>

<h2 id="samples">④ {n} 个已验证样本(点进去看完整编码;缩略图 = 原图 | 像素重建 | embeds PCA)</h2>
<div class="cards">{cards}</div>
"""
    open(os.path.join(OUT, "index.html"), "w").write(page(
        "Fast-dDrive SASD 数据集 v2 — 总览", body, n))


# ───────────────────────────────────────────────────────── sample pages ────
def write_sample(s, n):
    k, sid = s["k"], s["sid"]
    a = f"assets/{k:02d}"
    emb = s["emb"]
    checks_html = "".join(f"<tr><td><code>{E(nm)}</code></td><td>{v}</td></tr>"
                          for nm, v in s["checks"])
    secs_html = "".join(
        f"<tr><td>{b}</td><td><code>{E(sec)}</code></td><td>{pos}</td><td>{w:g}</td>"
        f"<td>{al:g}</td><td>{be:g}</td></tr>"
        for b, sec, pos, w, al, be in s["sec_rows"])
    origs = "".join(
        f'<div><img src="{a}/orig_{i}.jpg" loading="lazy" alt="{t}">'
        f'<p class="kv">image{i + 1} · {t}(原始 JPEG,1920×1280)</p></div>'
        for i, t in enumerate(["FRONT_LEFT 左前", "FRONT 正前", "FRONT_RIGHT 右前"]))
    prev_html = (f'<a href="sample_{k - 1:02d}.html">← 样本 {k - 1:02d}</a>'
                 if k > 0 else '<a href="index.html">← 总览</a>')
    next_html = (f'<a href="sample_{k + 1:02d}.html">样本 {k + 1:02d} →</a>'
                 if k < n - 1 else '<a href="index.html">回总览 →</a>')
    n_ok = sum(1 for _, v in s["checks"] if v == "✅")
    split_badge = ('<span class="badge val">val split</span>' if s["split"] == "val"
                   else '<span class="badge info">train split</span>')

    body = f"""
<div class="pn">{prev_html}{next_html}</div>
<h1>样本 {k:02d} {split_badge} <span class="badge ok">✅ {n_ok}/{len(s["checks"])} checks PASS</span></h1>
<p class="sub"><code>{E(sid)}</code> · v2 AR({s["split"]})shard {s["loc"]["shard"]}
row {s["loc"]["row"]} · L={s["L"]} · 答案区间 [{s["rs"]}, {s["re"]}) · {s["n_blocks"]} blocks
· 每图 {s["runs"]} merged tokens · pixel_values [{s["npix"]},1176] f16
· image_embeds [168,2048] bf16 · MASK 尾部 pad {s["mask_tail"]}</p>

<h2>1 · 原始输入:三张前视相机图</h2>
<div class="grid c3">{origs}</div>

<h2>2 · 像素 → 重建 → <span class="badge v2">v2</span> ViT embeds(模型实际吃的图像表示)</h2>
<p class="kv">每行一台相机:左=原图缩放;中=从数据集 <code>pixel_values</code> 反演重建(逐像素一致
→ 像素列正确);右=该图 56 个 <code>image_embeds</code>(8×7 merged-token 网格)的 top-3 PCA→RGB
——语义相近区域颜色相近、空间结构与图像对应,直观确认 embeds 编码的就是这张图。</p>
<img src="{a}/triptych__orig_recon_embedsPCA.png" alt="triptych">
<h3>embeds 数值验证:从存储像素重新过 frozen ViT(fp32 highest → bf16)</h3>
<div class="grid c2">
<div><img src="{a}/embeds_recompute_diff.png" alt="embeds diff"></div>
<div class="kv"><p>bitwise 一致 <b>{emb["bitwise_frac"]:.4%}</b>;全部 mismatch 都是舍入边界翻转
(≤1.5× 自身 bf16 ulp 或 |Δ|≤2e-3);max|Δ| = {emb["max_abs_diff"]:.2e};bf16 级全局 rel
{emb["global_rel"]:.2e}(仅诊断)。热图无空间结构 → 纯数值噪声而非内容差异。
真正的损坏(错图/错行)会让 bitwise 一致率崩到 ~0%。</p></div></div>

<h2>3 · 原始输入:文本 prompt(含数值自车历史)与训练目标</h2>
<details><summary>展开 prompt 原文(Stage-1 生成,navigation = {E(s["vd"]["nav"])})</summary>
<pre>{E(s["prompt"])}</pre></details>
<details><summary>展开 target:原始 gpt JSON(Stage-1)</summary><pre>{E(s["gpt_raw"])}</pre></details>
<h3>BEV 俯视图:prompt 里的历史 + 答案里编码的轨迹 vs 真值</h3>
<img src="{a}/bev_trajectory.png" alt="bev">
<p class="kv">蓝=prompt 中 7 个历史点(@0.5s);红 ×=答案 JSON 编码的 5 个 waypoint(@1s);
灰=tfrecord 真值 20 点(@4Hz)。红 × 落在灰线上 → 答案轨迹就是真值。fmb:
{E(s["vd"]["fmb_longitudinal"])} / {E(s["vd"]["fmb_lateral"])}。</p>

<h2>4 · 编码结果:token 序列</h2>
<img src="{a}/sequence_layout.png" alt="layout">
<details><summary>展开完整解码的 input_ids(占位符折叠;⟨…×N⟩ 表示 N 个连续相同 token)</summary>
<pre>{E(s["full"])}</pre></details>
<details open><summary>训练目标(labels 区间解码 = process_gpt 规范化后的答案)</summary>
<pre>{E(s["ans"])}</pre></details>
<h3>分 section 的块结构(权重 + Beta 噪声调度)</h3>
<table><tr><th>block</th><th>section</th><th>#token</th><th>loss 权重</th><th>α</th><th>β</th></tr>
{secs_html}</table>

<h2>5 · 位置编码与注意力 mask(模型的另两路输入)</h2>
<div class="grid c2"><div><img src="{a}/position_ids.png" alt="mrope"></div>
<div><img src="{a}/attn_mask.png" alt="mask"></div></div>

<h2>6 · 训练 step 实际看到的序列(在线加噪,seed 0)</h2>
<p class="kv">row 0(mdm 行)mask 了 {s["n_masked"]}/{s["n_resp"]} 个答案 token;
row 1(互补行)mask 了 {s["n_comp"]} 个(两行 mask 的 value 位置互补)。下面是两行
noisy 半边的答案区段:</p>
<details open><summary>row 0(mdm)</summary><pre>{E(s["noisy"])}</pre></details>
<details><summary>row 1(complementary)</summary><pre>{E(s["comp"])}</pre></details>

<h2>7 · 验证结果(v2 AR 记录 vs 从 raw 重跑整条链 + embeds 复算)</h2>
<table><tr><th>check</th><th>结果</th></tr>{checks_html}</table>
<div class="pn">{prev_html}{next_html}</div>
"""
    open(os.path.join(OUT, f"sample_{k:02d}.html"), "w").write(page(
        f"样本 {k:02d} · {sid}", body, n))


def write_readme():
    open(os.path.join(OUT, "README.md"), "w").write(
        "# Fast-dDrive SASD dataset **v2** review site\n\n"
        "Static, self-contained (relative paths only). Covers the v2 ArrayRecord format\n"
        "(13 fields + precomputed `image_embeds`), the raw→AR build chain, the v2 train\n"
        "feed path, and 12 round-2-verified examples (10 train + 2 val).\n\n"
        "Three ways to view from a Mac connected over SSH:\n\n"
        "1. **Port-forward (recommended)** — on the desktop:\n"
        "   `bash serve.sh` (serves on :8890), then on the Mac:\n"
        "   `ssh -L 8890:localhost:8890 <desktop>` and open http://localhost:8890\n"
        "2. **Copy to Mac** — `scp -r <desktop>:" + OUT + " ~/Desktop/` then open `index.html`.\n"
        "3. **VS Code Remote** — Live Server on `index.html` (auto port-forward).\n\n"
        "Regenerate after a new verify run (`NT=10 NV=2 bash "
        "jax_ddrive/scripts/verify_ar_round2.sh`):\n"
        "`PYTHONPATH=jax_ddrive ddrive-python jax_ddrive/scripts/make_dataset_website.py`\n")
    sv = os.path.join(OUT, "serve.sh")
    open(sv, "w").write('#!/usr/bin/env bash\ncd "$(dirname "$0")"\n'
                        'exec python3 -m http.server "${1:-8890}"\n')
    os.chmod(sv, 0o755)


if __name__ == "__main__":
    main()
