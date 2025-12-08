#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 {1p|10p} OUTPUT_DIR [CONCURRENCY]"
  exit 1
fi

SPLIT="$1"
OUTPUT_DIR="$2"
CONCURRENCY="${3:-4}"

HF_PATH="UCSC-VLAA/Recap-DataComp-1B"
HF_REVISION="main"

case "$SPLIT" in
  1p|10p)
    ;;
  *)
    echo "first arg must be '1p' or '10p'"
    exit 1
    ;;
esac

mkdir -p "$OUTPUT_DIR"

SUCCESS_LIST="$OUTPUT_DIR/downloaded_ok.txt"
: > "$SUCCESS_LIST"

download_one() {
  local url="$1"
  local dest="$2"
  local idx="$3"
  echo "[$idx] START $(basename "$dest")"
  if wget -c --tries=5 --timeout=30 -O "$dest" "$url"; then
    echo "[$idx] DONE  $(basename "$dest")"
    echo "$dest" >> "$SUCCESS_LIST"
  else
    echo "[$idx] FAIL  $(basename "$dest")" >&2
  fi
}

mapfile -t entries < <(
python - "$HF_PATH" "$HF_REVISION" << 'EOF'
import sys
from huggingface_hub import list_repo_files, hf_hub_url

repo_id = sys.argv[1]
revision = sys.argv[2]

files = list_repo_files(repo_id, revision=revision, repo_type="dataset")
for f in files:
    if f.startswith("data/train_data/") and f.endswith(".parquet"):
        url = hf_hub_url(repo_id=repo_id, filename=f, revision=revision, repo_type="dataset")
        print(url, f)
EOF
)

i=0

for line in "${entries[@]}"; do
  read -r url relpath <<< "$line" || continue
  [[ -z "${url:-}" || -z "${relpath:-}" ]] && continue
  out_path="$OUTPUT_DIR/$relpath"
  mkdir -p "$(dirname "$out_path")"
  download_one "$url" "$out_path" "$i" &
  i=$((i+1))
  if (( i % CONCURRENCY == 0 )); then
    wait
  fi
done

wait

success_count=$(wc -l < "$SUCCESS_LIST" 2>/dev/null || echo 0)
echo "Total downloaded: $success_count file(s)."
echo "Success list saved to: $SUCCESS_LIST"
