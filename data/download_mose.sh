#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$SCRIPT_DIR/mose"

# Accept repo ids or full URLs via args or env
MOSE1_REPO_RAW="${1:-${MOSE1_REPO:-FudanCVL/MOSE}}"
MOSE2_REPO_RAW="${2:-${MOSE2_REPO:-FudanCVL/MOSEv2}}"

normalize_repo_id() {
  local in="$1"
  local out="$in"
  if [[ "$out" == https://huggingface.co/datasets/* ]]; then
    out="${out#https://huggingface.co/datasets/}"
    out="${out%%/tree/*}"
  fi
  echo "$out"
}

MOSE1_REPO="$(normalize_repo_id "$MOSE1_REPO_RAW")"
MOSE2_REPO="$(normalize_repo_id "$MOSE2_REPO_RAW")"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Please install it: pip install huggingface_hub" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/mose1" "$OUTPUT_ROOT/mose2"

download_hf_repo() {
  local repo_id="$1"
  local dest_dir="$2"
  echo "Downloading Hugging Face dataset repo: $repo_id -> $dest_dir"
  mkdir -p "$dest_dir"
  huggingface-cli download "$repo_id" \
    --repo-type dataset \
    --include "*" \
    --local-dir "$dest_dir" \
    --local-dir-use-symlinks False
}

unzip_and_cleanup() {
  local dest_dir="$1"
  echo "Unzipping any .zip archives in $dest_dir and removing them afterward"
  find "$dest_dir" -type f -name "*.zip" -print0 | while IFS= read -r -d '' z; do
    unzip -o "$z" -d "$(dirname "$z")"
    rm -f "$z"
  done
}

# MOSE 1 (train/val)
download_hf_repo "$MOSE1_REPO" "$OUTPUT_ROOT/mose1"
unzip_and_cleanup "$OUTPUT_ROOT/mose1"

# MOSE 2 (train/val)
download_hf_repo "$MOSE2_REPO" "$OUTPUT_ROOT/mose2"
unzip_and_cleanup "$OUTPUT_ROOT/mose2"
