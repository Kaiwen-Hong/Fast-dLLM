# Fast-dDrive SASD dataset **v2** review site

Static, self-contained (relative paths only). Covers the v2 ArrayRecord format
(13 fields + precomputed `image_embeds`), the raw→AR build chain, the v2 train
feed path, and 12 round-2-verified examples (10 train + 2 val).

Three ways to view from a Mac connected over SSH:

1. **Port-forward (recommended)** — on the desktop:
   `bash serve.sh` (serves on :8890), then on the Mac:
   `ssh -L 8890:localhost:8890 <desktop>` and open http://localhost:8890
2. **Copy to Mac** — `scp -r <desktop>:/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/visualizations ~/Desktop/` then open `index.html`.
3. **VS Code Remote** — Live Server on `index.html` (auto port-forward).

Regenerate after a new verify run (`NT=10 NV=2 bash jax_ddrive/scripts/verify_ar_round2.sh`):
`PYTHONPATH=jax_ddrive ddrive-python jax_ddrive/scripts/make_dataset_website.py`
