#!/usr/bin/env bash
#
# Stage 2 Training Script
# Usage:
#   bash stage2/scripts/train_memory.sh \
#     SAM3_CKPT=sam3_checkpoints/sam3.pt \
#     DATA_PATH=data/sa-v/extracted_frames \
#     OUTPUT=output/stage2_checkpoints \
#     BATCH_SIZE=4 \
#     GPUS=8

set -euo pipefail

# Allow KEY=VALUE overrides passed after the script name.
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    SAM3_CKPT=*|DATA_PATH=*|OUTPUT=*|BATCH_SIZE=*|EPOCHS=*|LR=*|GPUS=*|MASTER_PORT=*|NNODES=*|NODE_RANK=*|RDZV_BACKEND=*|RDZV_ENDPOINT=*)
      key=${arg%%=*}
      value=${arg#*=}
      printf -v "$key" '%s' "$value"
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done
set -- "${EXTRA_ARGS[@]}"

# Default values
SAM3_CKPT="${SAM3_CKPT:-sam3_checkpoints/sam3.pt}"
DATA_PATH="${DATA_PATH:-data/sa-v/extracted_frames}"
OUTPUT="${OUTPUT:-output/stage2_checkpoints}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-4}"
GPUS="${GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:${MASTER_PORT}}"

# Construct torchrun arguments
TORCHRUN_ARGS=(--nproc_per_node "${GPUS}")
if [ "${NNODES}" -gt 1 ]; then
  TORCHRUN_ARGS+=(--nnodes "${NNODES}" --node_rank "${NODE_RANK}" --rdzv_backend "${RDZV_BACKEND}" --rdzv_endpoint "${RDZV_ENDPOINT}")
else
  TORCHRUN_ARGS+=(--nnodes 1 --master_port "${MASTER_PORT}")
fi

# Construct python script arguments
PY_ARGS=(
  --sam3_checkpoint "${SAM3_CKPT}"
  --data_path "${DATA_PATH}"
  --output_dir "${OUTPUT}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
)

echo "Launching training with ${GPUS} GPUs..."
echo "Output dir: ${OUTPUT}"

PYTHONPATH=. torchrun "${TORCHRUN_ARGS[@]}" \
  stage2/train_memory_stage2.py \
  "${PY_ARGS[@]}" \
  "$@"
