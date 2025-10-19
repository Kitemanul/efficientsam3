#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$SCRIPT_DIR/ytvos"

# Allow overriding URL via arg or env (prefer arg)
YTVOS_URL="${1:-${YTVOS_URL:-https://drive.google.com/drive/folders/1XwjQ-eysmOb7JdmJAwfVOBZX-aMbHccC}}"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown not found. Please install it first: pip install gdown" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

echo "Downloading YouTube-VOS folder: $YTVOS_URL -> $OUTPUT_ROOT"
gdown --folder --remaining-ok -O "$OUTPUT_ROOT" "$YTVOS_URL"

echo "Unzipping any .zip archives under $OUTPUT_ROOT and removing them afterward"
find "$OUTPUT_ROOT" -type f -name "*.zip" -print0 | while IFS= read -r -d '' z; do
  unzip -o "$z" -d "$(dirname "$z")"
  rm -f "$z"
done