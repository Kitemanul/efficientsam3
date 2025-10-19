#!/usr/bin/env bash
set -euo pipefail

# Download SA-1B dataset archives listed in a TSV file with columns:
# file_name<TAB>cdn_link
#
# Usage:
#   ./download_sa1b.sh [INPUT_TSV] [OUTPUT_DIR] [CONCURRENCY]
#
# Defaults:
#   INPUT_TSV   = data/sa-1b.txt (next to this script)
#   OUTPUT_DIR  = data/sa-1b/    (next to this script)
#   CONCURRENCY = 1               (number of parallel downloads)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_TSV="${1:-"$SCRIPT_DIR/sa-1b.txt"}"
OUTPUT_DIR="${2:-"$SCRIPT_DIR/sa-1b"}"
CONCURRENCY="${3:-1}"

if [[ ! -f "$INPUT_TSV" ]]; then
  echo "Input file not found: $INPUT_TSV" >&2
  exit 1
fi

if ! [[ "$CONCURRENCY" =~ ^[0-9]+$ ]] || [[ "$CONCURRENCY" -lt 1 ]]; then
  echo "Invalid concurrency: $CONCURRENCY (must be a positive integer)" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Downloading to: $OUTPUT_DIR"
echo "Reading list from: $INPUT_TSV"
echo "Concurrency: $CONCURRENCY"

trap 'echo "Interrupted. Waiting for child processes..."; trap - INT; kill 0 2>/dev/null || true; wait || true' INT

download_one() {
  local file_name="$1"
  local url="$2"

  if [[ -z "$file_name" || -z "$url" ]]; then
    return 0
  fi

  local dest="$OUTPUT_DIR/$file_name"
  echo "[START] $file_name"
  # Resume partial downloads, retry a few times, and set timeouts
  if wget -c --tries=5 --timeout=30 -O "$dest" "$url"; then
    echo "[DONE]  $file_name"
  else
    echo "[FAIL]  $file_name" >&2
    return 1
  fi
}

# Skip header, ensure at least 2 fields, and pass tab-delimited pairs to the loop
err_count=0
idx=0
while IFS=$'\t' read -r file_name url _rest; do
  # Skip empty lines
  [[ -z "${file_name:-}" && -z "${url:-}" ]] && continue
  # Skip header if present
  if [[ "$file_name" == "file_name" && "$url" == "cdn_link" ]]; then
    continue
  fi

  # Detach background job from stdin to avoid competing with the read loop
  download_one "$file_name" "$url" </dev/null &
  (( idx++ ))
  # Limit number of concurrent background jobs
  if (( idx % CONCURRENCY == 0 )); then
    wait || err_count=$((err_count+1))
  fi
done < <(awk -F '\t' 'NF>=2 {print $1 "\t" $2}' "$INPUT_TSV")

# Wait for remaining background jobs
wait || err_count=$((err_count+1))

if (( err_count > 0 )); then
  echo "Completed with $err_count error group(s)." >&2
  exit 2
fi

echo "All downloads completed successfully."


