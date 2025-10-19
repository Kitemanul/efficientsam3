#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$SCRIPT_DIR/lvis"

mkdir -p "$OUTPUT_ROOT/annotations" "$OUTPUT_ROOT/images"

download_and_extract() {
  local url="$1"
  local dest_dir="$2"
  local filename
  filename="$(basename "$url")"

  echo "Downloading $filename -> $dest_dir"
  mkdir -p "$dest_dir"
  wget -nc -O "$dest_dir/$filename" "$url"

  echo "Unzipping $filename in $dest_dir"
  unzip -o "$dest_dir/$filename" -d "$dest_dir"

  echo "Removing archive $filename"
  rm -f "$dest_dir/$filename"
}

# LVIS annotations (v1)
download_and_extract "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip" "$OUTPUT_ROOT/annotations"
download_and_extract "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip" "$OUTPUT_ROOT/annotations"

# COCO images used by LVIS
download_and_extract "http://images.cocodataset.org/zips/train2017.zip" "$OUTPUT_ROOT/images"
download_and_extract "http://images.cocodataset.org/zips/val2017.zip" "$OUTPUT_ROOT/images"