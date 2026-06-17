#!/usr/bin/env bash
# Pack the Fast-dDrive code and upload it to GCS for the internal side to pull.
#
# As of 2026-06-16 the code is ONE repo: `maxtext-dlm-fork/` + `jax_ddrive/` are both sub-trees of
# the Fast-dLLM repo, so a SINGLE git commit describes the whole bundle. Each upload writes a
# MANIFEST.json recording that commit (+ a dirty flag) so a deployed bundle can be mapped back to
# exact source — the timestamp is kept only as a unique, sortable object name.
#
#   bundle   -> gs://<bucket>/code/fastddrive-<TS>-<sha7>[-dirty].tgz
#               extracts to  fastddrive-<TS>-<sha7>/{maxtext-dlm-fork, jax_ddrive, MANIFEST.json}
#   pointer  -> gs://<bucket>/code/fastddrive-LATEST.txt              (names the newest .tgz)
#   manifest -> gs://<bucket>/code/<NAME>.MANIFEST.json  +  fastddrive-LATEST-MANIFEST.json
#
# Run from anywhere (repo root is derived from this script's own location):
#   bash jax_ddrive/scripts/upload_code_to_gcs.sh
#
# Re-run on every code change. COMMIT FIRST if you want the recorded SHA to fully describe the
# bundle — an uncommitted working tree is published with a loud warning and a `-dirty` marker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"          # jax_ddrive/scripts/ -> Fast-dLLM
BKT="${BKT:-gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd}"
cd "$REPO"

# the sub-trees that make up the shipped codebase
#   maxtext-dlm-fork = training + B1 export + B2 self-contained inference (production)
#   jax_ddrive       = offline input-prep + NNX reference + docs
#   fast_ddrive      = WOD-E2E converter (convert_wod_e2e.py) + official metric
#                      (evaluate_waymo_metrics.py) — needed by STEP 3 (data + val parity)
PACK_DIRS=(maxtext-dlm-fork jax_ddrive fast_ddrive)
for d in "${PACK_DIRS[@]}"; do [ -d "$REPO/$d" ] || { echo "ERROR: $REPO/$d not found" >&2; exit 1; }; done

# ---- git provenance (single repo) ----
SHA="$(git rev-parse HEAD)"
SHA7="$(git rev-parse --short=7 HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
REMOTE="$(git remote get-url origin 2>/dev/null || echo NONE)"
if [ -n "$(git status --porcelain -- "${PACK_DIRS[@]}")" ]; then
  DIRTY=true
  echo "WARNING: uncommitted changes in ${PACK_DIRS[*]} — bundle marked '-dirty';" >&2
  echo "         commit first if you want SHA ${SHA7} to fully describe this bundle." >&2
else
  DIRTY=false
fi

TS="$(date +%Y%m%d_%H%M%S)"
SUFFIX=""; [ "$DIRTY" = true ] && SUFFIX="-dirty"
NAME="fastddrive-${TS}-${SHA7}${SUFFIX}"
TGZ="/tmp/${NAME}.tgz"
TMPD="$(mktemp -d)"

# ---- MANIFEST.json (provenance the internal side can pin / verify against GitHub) ----
cat > "$TMPD/MANIFEST.json" <<EOF
{
  "bundle": "${NAME}.tgz",
  "timestamp": "${TS}",
  "git_commit": "${SHA}",
  "git_commit_short": "${SHA7}",
  "git_branch": "${BRANCH}",
  "git_remote": "${REMOTE}",
  "dirty": ${DIRTY},
  "contents": ["maxtext-dlm-fork/", "jax_ddrive/", "fast_ddrive/"],
  "note": "maxtext-dlm-fork was consolidated into the Fast-dLLM repo on 2026-06-16 (from a standalone repo @ da8c92b; full history backup: /home/kaiwen/data/fast-ddrive/maxtext-dlm-fork-history-2026-06-16.bundle). One git_commit describes the whole bundle; if dirty=true the working tree had uncommitted edits beyond that commit."
}
EOF

# ---- pack: the two sub-trees (working tree) + MANIFEST.json at the bundle root ----
# --transform prefixes every member with fastddrive-<TS>-<sha7>/ so extraction is self-naming.
tar czf "$TGZ" \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' \
  --exclude='.pytest_cache' --exclude='.claude' --exclude='jax_ddrive/visualizations' \
  --transform "s,^,${NAME}/," \
  -C "$REPO" "${PACK_DIRS[@]}" \
  -C "$TMPD" MANIFEST.json

SZ="$(du -h "$TGZ" | cut -f1)"

# ---- upload: tarball + per-bundle manifest + LATEST pointers ----
gsutil cp "$TGZ" "$BKT/code/${NAME}.tgz"
gsutil cp "$TMPD/MANIFEST.json" "$BKT/code/${NAME}.MANIFEST.json"
echo "${NAME}.tgz" | gsutil cp - "$BKT/code/fastddrive-LATEST.txt"
gsutil cp "$TMPD/MANIFEST.json" "$BKT/code/fastddrive-LATEST-MANIFEST.json"
rm -rf "$TGZ" "$TMPD"

echo "UPLOADED ${BKT}/code/${NAME}.tgz  (${SZ})  dirty=${DIRTY}"
echo "  commit  : ${SHA} (${BRANCH})"
echo "  LATEST  -> ${BKT}/code/fastddrive-LATEST.txt = ${NAME}.tgz"
echo "  manifest-> ${BKT}/code/fastddrive-LATEST-MANIFEST.json"
echo "  internal extract target: .../third_party/${NAME}"
