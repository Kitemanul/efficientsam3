#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$SCRIPT_DIR/lvos"

# Allow overriding URLs via args or env (prefer args)
LVOS_TRAIN_URL="${1:-${LVOS_TRAIN_URL:-https://drive.google.com/file/d/1-ehpl5s0Fd14WwtT-GmWtIWa_BxZl9D6/view}}"
LVOS_VAL_URL="${2:-${LVOS_VAL_URL:-https://drive.google.com/file/d/17Hwc__6i2rpF5e2s5OPqoywNxG5bzlcO/view}}"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown not found. Please install it first: pip install gdown" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/train" "$OUTPUT_ROOT/val"

download_file() {
  local url="$1"
  local dest_dir="$2"
  echo "Downloading: $url -> $dest_dir"
  mkdir -p "$dest_dir"
  pushd "$dest_dir" >/dev/null
  gdown --fuzzy --remaining-ok "$url"
  popd >/dev/null
}

unzip_and_cleanup() {
  local dest_dir="$1"
  echo "Unzipping any .zip archives in $dest_dir and removing them afterward"
  find "$dest_dir" -type f -name "*.zip" -print0 | while IFS= read -r -d '' z; do
    unzip -o "$z" -d "$(dirname "$z")"
    rm -f "$z"
  done
}

# LVOS train
download_file "$LVOS_TRAIN_URL" "$OUTPUT_ROOT/train"
unzip_and_cleanup "$OUTPUT_ROOT/train"

# LVOS val
download_file "$LVOS_VAL_URL" "$OUTPUT_ROOT/val"
unzip_and_cleanup "$OUTPUT_ROOT/val"