#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$SCRIPT_DIR/davis"

mkdir -p "$OUTPUT_ROOT/2016" \
         "$OUTPUT_ROOT/2017/trainval" \
         "$OUTPUT_ROOT/2017/unsupervised" \

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

# DAVIS 2016
download_and_extract "https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip" "$OUTPUT_ROOT/2016"

# DAVIS 2017 (Supervised, Full-Resolution)
download_and_extract "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip" "$OUTPUT_ROOT/2017/trainval"

# DAVIS 2017 (Unsupervised, Full-Resolution)
download_and_extract "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-Full-Resolution.zip" "$OUTPUT_ROOT/2017/unsupervised"

