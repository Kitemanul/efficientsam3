#!/bin/bash
# --------------------------------------------------------
# Save Video Embeddings for Stage 2
# --------------------------------------------------------
# Pre-compute trunk features from SAM3 backbone.
# Supports multi-GPU via torchrun.
#
# Usage:
#   # Single GPU
#   bash stage2/scripts/save_embeddings.sh \
#     DATA_DIR=data/sa-v/extracted_frames \
#     OUTPUT=output/stage2_embeddings
#
#   # Multi-GPU (4 GPUs)
#   bash stage2/scripts/save_embeddings.sh \
#     DATA_DIR=data/sa-v/extracted_frames \
#     OUTPUT=output/stage2_embeddings \
#     GPUS=4

set -e

# Default values
DATA_DIR="${DATA_DIR:-data/sa-v/extracted_frames}"
OUTPUT="${OUTPUT:-output/stage2_embeddings}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CHECKPOINT="${CHECKPOINT:-}"
GPUS="${GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Parse command line arguments (KEY=VALUE format)
for arg in "$@"; do
  case $arg in
    DATA_DIR=*)
      DATA_DIR="${arg#*=}"
      ;;
    OUTPUT=*)
      OUTPUT="${arg#*=}"
      ;;
    BATCH_SIZE=*)
      BATCH_SIZE="${arg#*=}"
      ;;
    CHECKPOINT=*)
      CHECKPOINT="${arg#*=}"
      ;;
    GPUS=*)
      GPUS="${arg#*=}"
      ;;
    MASTER_PORT=*)
      MASTER_PORT="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: bash save_embeddings.sh DATA_DIR=path OUTPUT=path BATCH_SIZE=8 GPUS=1"
      exit 1
      ;;
  esac
done

echo "=============================================="
echo "Stage 2: Save Video Embeddings"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT"
echo "Batch size: $BATCH_SIZE"
echo "GPUs: $GPUS"
echo "=============================================="

# Build base arguments
ARGS="--data_dir $DATA_DIR --output_dir $OUTPUT --batch_size $BATCH_SIZE"

# Add checkpoint if specified
if [ -n "$CHECKPOINT" ]; then
  ARGS="$ARGS --checkpoint $CHECKPOINT"
fi

# Build command based on number of GPUs
if [ "$GPUS" -gt 1 ]; then
  CMD="torchrun --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    stage2/save_video_embeddings_stage2.py $ARGS"
else
  CMD="python stage2/save_video_embeddings_stage2.py $ARGS"
fi

# Run
echo "Running: $CMD"
eval $CMD

echo "Done!"
