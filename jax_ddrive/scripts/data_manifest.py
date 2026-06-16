#!/usr/bin/env python3
"""Build a DATA_MANIFEST.json for a data artifact (GCS prefix or local dir) so a run can pin /
verify EXACTLY which data it consumed — closing the "named GCS paths are overwritten in place,
no version" provenance gap.

- GCS  : reads each object's crc32c + size + generation from `gsutil ls -L` (**no download** — cheap
         even for the 369 GB full dataset).
- local: streams sha256 per file (capped — see build_local_manifest).

A single rolled-up `digest` (sha256 over the sorted "path<TAB>hash<TAB>size" lines) identifies the
whole artifact. Record that digest in `validation_log.jsonl` (a `data_provenance` event) alongside the
code MANIFEST's git_commit, and a run becomes fully reconstructable: {code git_sha} x {data digests}.

Usage:
  python jax_ddrive/scripts/data_manifest.py gs://bucket/path/to/artifact              # print JSON
  python jax_ddrive/scripts/data_manifest.py <local_dir> --out DATA_MANIFEST.json
  python jax_ddrive/scripts/data_manifest.py gs://.../artifact --upload                # write + gsutil cp next to it
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def _git_sha(start):
    try:
        return subprocess.check_output(
            ["git", "-C", start, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def rolled_digest(files, key):
    """sha256 over the sorted 'path<TAB><key-value><TAB>size' lines — one digest for the artifact."""
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f['path']}\t{f.get(key, '')}\t{f['size']}\n".encode())
    return h.hexdigest()


def gcs_files(prefix):
    """[{path, crc32c, size, generation}] for every object under a gs:// prefix (no download)."""
    prefix = prefix.rstrip("/")
    out = subprocess.check_output(["gsutil", "ls", "-L", prefix + "/**"]).decode(errors="replace")
    files, cur = [], None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("gs://") and s.endswith(":"):
            if cur and "crc32c" in cur:
                files.append(cur)
            cur = {"path": s[:-1]}
        elif cur is not None:
            if s.startswith("Content-Length:"):
                cur["size"] = int(s.split(":", 1)[1].strip())
            elif s.startswith("Hash (crc32c):"):
                cur["crc32c"] = s.split(":", 1)[1].strip()
            elif s.startswith("Generation:"):
                cur["generation"] = s.split(":", 1)[1].strip()
    if cur and "crc32c" in cur:
        files.append(cur)
    for f in files:  # store paths relative to the prefix
        f["path"] = f["path"].replace(prefix + "/", "", 1)
        f.setdefault("size", 0)
    return sorted(files, key=lambda f: f["path"])


def build_local_manifest(root, hash_cap_bytes=8 << 30, builder_git_sha=None):
    """Manifest dict for a local dir. Per-file sha256 unless total > cap (then sizes only + a note —
    run this tool on the GCS copy for crc32c hashes). Importable by the dataset builder."""
    root = os.path.abspath(root)
    entries = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn == "DATA_MANIFEST.json":
                continue
            fp = os.path.join(dp, fn)
            entries.append((os.path.relpath(fp, root), fp, os.path.getsize(fp)))
    entries.sort()
    total = sum(e[2] for e in entries)
    do_hash = total <= hash_cap_bytes
    files = []
    for rel, fp, sz in entries:
        f = {"path": rel, "size": sz}
        if do_hash:
            f["sha256"] = _sha256_file(fp)
        files.append(f)
    man = {
        "artifact": root,
        "source": "local",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hash_kind": "sha256" if do_hash else "size-only",
        "builder_git_sha": builder_git_sha or _git_sha(root),
        "num_files": len(files),
        "total_bytes": total,
        "digest": rolled_digest(files, "sha256" if do_hash else "size"),
        "files": files,
    }
    if not do_hash:
        man["note"] = (
            f"total {total} B > cap {hash_cap_bytes} B; per-file content hashes skipped — "
            "run data_manifest.py on the GCS copy for crc32c hashes."
        )
    return man


def main():
    ap = argparse.ArgumentParser(description="Build a DATA_MANIFEST.json for a data artifact.")
    ap.add_argument("artifact", help="gs://... prefix or local dir/file")
    ap.add_argument("--out", help="write JSON here (default: stdout)")
    ap.add_argument("--upload", action="store_true",
                    help="also gsutil cp the manifest to <artifact>/DATA_MANIFEST.json (gcs only)")
    args = ap.parse_args()

    is_gcs = args.artifact.startswith("gs://")
    if is_gcs:
        files = gcs_files(args.artifact)
        manifest = {
            "artifact": args.artifact.rstrip("/"),
            "source": "gcs",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "hash_kind": "crc32c",
            "builder_git_sha": _git_sha(os.path.dirname(os.path.abspath(__file__))),
            "num_files": len(files),
            "total_bytes": sum(f["size"] for f in files),
            "digest": rolled_digest(files, "crc32c"),
            "files": files,
        }
    else:
        manifest = build_local_manifest(args.artifact)

    text = json.dumps(manifest, indent=2)
    if args.out:
        open(args.out, "w").write(text)
        print(f"wrote {args.out}  digest={manifest['digest'][:16]}... files={manifest['num_files']}")
    else:
        print(text)

    if args.upload:
        if not is_gcs:
            sys.exit("--upload is only for gs:// artifacts")
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as t:
            t.write(text)
            tmp = t.name
        dest = args.artifact.rstrip("/") + "/DATA_MANIFEST.json"
        subprocess.check_call(["gsutil", "cp", tmp, dest])
        os.unlink(tmp)
        print(f"uploaded -> {dest}")


if __name__ == "__main__":
    main()
