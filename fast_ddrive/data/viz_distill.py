#!/usr/bin/env python3
"""Render a self-contained HTML to eyeball teacher-distilled labels vs the pseudo labels.

For each sample: the 3 front-view frames (left/center/right), a BEV plot of the
real-GT trajectory, and a side-by-side of pseudo vs distilled
critical_objects / explanation / future_meta_behavior. Images are downscaled and
base64-embedded so the HTML opens from anywhere.

Usage::

    python fast_ddrive/data/viz_distill.py \
        --distilled_json /home/kaiwen/data/fast-ddrive/train/train_targets_distilled_10.json \
        --orig_json      /home/kaiwen/data/fast-ddrive/train/train_targets.json \
        --image_root     /home/kaiwen/data/fast-ddrive/train/train_images \
        --out_dir        /home/kaiwen/data/fast-ddrive/train/distill_viz_10
"""
import argparse
import base64
import html
import io
import json
import os
import re

from PIL import Image


def strip_markers(s):
    return str(s).replace("<|mdm_start|>", "").replace("<|mdm_end|>", "").strip()


def img_b64(path, width=420):
    try:
        im = Image.open(path).convert("RGB")
        if im.width > width:
            im = im.resize((width, int(im.height * width / im.width)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        return ""


def traj_points(traj_str):
    return [(float(x), float(y)) for x, y in re.findall(r'\[([+-][\d.]+),\s*([+-][\d.]+)\]', strip_markers(traj_str))]


def traj_svg(points, w=200, h=260, pad=24):
    """BEV: x=forward (up), y=lateral (right=+x on screen). Ego at origin (bottom-center)."""
    if not points:
        return "<div class='muted'>no trajectory</div>"
    xs = [0.0] + [p[0] for p in points]   # forward
    ys = [0.0] + [p[1] for p in points]   # lateral
    fmax = max(max(xs), 1.0)
    lat = max(max(abs(min(ys)), abs(max(ys))), 2.0)
    def sx(y):  # lateral -> screen x (centered)
        return pad + (w - 2 * pad) * (y + lat) / (2 * lat)
    def sy(x):  # forward -> screen y (up)
        return (h - pad) - (h - 2 * pad) * (x / fmax)
    pts = list(zip(xs, ys))
    poly = " ".join(f"{sx(y):.1f},{sy(x):.1f}" for x, y in pts)
    dots = "".join(f"<circle cx='{sx(y):.1f}' cy='{sy(x):.1f}' r='3' fill='#2563eb'/>" for x, y in pts[1:])
    ego = f"<circle cx='{sx(0):.1f}' cy='{sy(0):.1f}' r='4' fill='#dc2626'/>"
    axis = (f"<line x1='{w/2}' y1='{pad}' x2='{w/2}' y2='{h-pad}' stroke='#e5e7eb'/>"
            f"<line x1='{pad}' y1='{h-pad}' x2='{w-pad}' y2='{h-pad}' stroke='#e5e7eb'/>")
    return (f"<svg width='{w}' height='{h}' style='background:#fafafa;border:1px solid #eee'>"
            f"{axis}<polyline points='{poly}' fill='none' stroke='#93c5fd' stroke-width='2'/>"
            f"{dots}{ego}"
            f"<text x='4' y='14' font-size='10' fill='#888'>fwd {fmax:.0f}m</text></svg>")


def co_html(co, label):
    if not isinstance(co, dict):
        return f"<span class='muted'>{label}: —</span>"
    yes = [k for k, v in co.items() if str(v).lower().startswith("y")]
    chips = "".join(f"<span class='yes'>{html.escape(k)}</span>" for k in yes) or "<span class='muted'>all no</span>"
    return f"<div><b>{label}</b> {chips}</div>"


def fmb_str(fmb):
    if not isinstance(fmb, dict):
        return "—"
    return f"long=<b>{html.escape(strip_markers(fmb.get('longitudinal','')))}</b>, lat=<b>{html.escape(strip_markers(fmb.get('lateral','')))}</b>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distilled_json", required=True)
    ap.add_argument("--orig_json", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max", type=int, default=-1)
    args = ap.parse_args()

    orig = {s["sample_id"]: s for s in json.load(open(args.orig_json))}
    dist = json.load(open(args.distilled_json))
    if args.max > 0:
        dist = dist[: args.max]
    os.makedirs(args.out_dir, exist_ok=True)

    cards = []
    for d in dist:
        sid = d["sample_id"]
        n = json.loads(d["conversations"][1]["value"])
        o = json.loads(orig[sid]["conversations"][1]["value"]) if sid in orig else {}
        imgs = "".join(
            f"<figure><img src='{img_b64(os.path.join(args.image_root, rel))}'/>"
            f"<figcaption>{html.escape(os.path.basename(rel))}</figcaption></figure>"
            for rel in d.get("image", [])
        )
        svg = traj_svg(traj_points(n.get("trajectory", "")))
        card = f"""
        <div class="card">
          <div class="hdr"><b>{html.escape(sid)}</b> &nbsp;·&nbsp; nav: {html.escape(str(d.get('navigation_command','')))}</div>
          <div class="imgs">{imgs}</div>
          <div class="cols">
            <div class="bev"><div class="lbl">GT trajectory (BEV)</div>{svg}</div>
            <div class="labels">
              <div class="row">
                <div class="pseudo"><div class="tag tag-p">PSEUDO</div>
                  {co_html(o.get('critical_objects'), 'critical_objects')}
                  <div class="exp">{html.escape(strip_markers(o.get('explanation','')))}</div>
                  <div class="fmb">fmb: {fmb_str(o.get('future_meta_behavior'))}</div>
                </div>
                <div class="distilled"><div class="tag tag-d">DISTILLED</div>
                  {co_html(n.get('critical_objects'), 'critical_objects')}
                  <div class="exp">{html.escape(strip_markers(n.get('explanation','')))}</div>
                  <div class="fmb">fmb: {fmb_str(n.get('future_meta_behavior'))} <span class='muted'>(long=GT, lat=teacher)</span></div>
                </div>
              </div>
            </div>
          </div>
        </div>"""
        cards.append(card)

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Distill viz ({len(dist)})</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;background:#f3f4f6;color:#111}}
 h1{{font-size:18px}} .card{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;margin:0 0 18px;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
 .hdr{{font-size:13px;color:#374151;margin-bottom:8px}}
 .imgs{{display:flex;gap:8px}} .imgs figure{{margin:0}} .imgs img{{width:100%;border-radius:6px;display:block}} figcaption{{font-size:10px;color:#9ca3af;text-align:center}}
 .cols{{display:flex;gap:14px;margin-top:10px;align-items:flex-start}} .bev .lbl{{font-size:11px;color:#6b7280;margin-bottom:4px}}
 .row{{display:flex;gap:12px}} .pseudo,.distilled{{flex:1;border:1px solid #eee;border-radius:8px;padding:10px;font-size:12.5px;line-height:1.45}}
 .pseudo{{background:#fff7ed}} .distilled{{background:#ecfdf5}}
 .tag{{display:inline-block;font-size:10px;font-weight:700;padding:1px 7px;border-radius:10px;margin-bottom:6px}}
 .tag-p{{background:#fed7aa;color:#9a3412}} .tag-d{{background:#a7f3d0;color:#065f46}}
 .yes{{display:inline-block;background:#fee2e2;color:#991b1b;border-radius:6px;padding:1px 6px;margin:0 3px 3px 0;font-size:11px}}
 .exp{{margin:6px 0;color:#374151}} .fmb{{color:#1f2937}} .muted{{color:#9ca3af}}
</style></head><body>
<h1>Teacher-distill visualization — {len(dist)} samples</h1>
<p class="muted">Left=pseudo (converter) · Right=distilled (teacher CO/explanation/lateral + GT-derived longitudinal). Trajectory = real GT, unchanged.</p>
{''.join(cards)}
</body></html>"""

    out = os.path.join(args.out_dir, "index.html")
    with open(out, "w") as f:
        f.write(page)
    print(f"[viz] wrote {len(dist)} samples -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    print("VIZ_DONE")


if __name__ == "__main__":
    main()
