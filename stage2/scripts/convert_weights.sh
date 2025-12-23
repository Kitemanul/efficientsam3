#!/bin/bash
# --------------------------------------------------------
# Convert Stage 2 Weights
# --------------------------------------------------------
# Merge trained memory modules with SAM3 checkpoint.
#
# Usage:
#   bash stage2/scripts/convert_weights.sh \
#     STAGE2_WEIGHTS=output/stage2_memory/ckpt_best.pth \
#     SAM3_WEIGHTS=sam3_checkpoints/sam3.pt \
#     OUTPUT=output/efficient_sam3_memory.pt

set -e

# Default values
STAGE2_WEIGHTS="${STAGE2_WEIGHTS:-output/stage2_memory/ckpt_best.pth}"
SAM3_WEIGHTS="${SAM3_WEIGHTS:-}"
OUTPUT="${OUTPUT:-output/efficient_sam3_memory.pt}"

# Parse command line arguments (KEY=VALUE format)
for arg in "$@"; do
  case $arg in
    STAGE2_WEIGHTS=*)
      STAGE2_WEIGHTS="${arg#*=}"
      ;;
    SAM3_WEIGHTS=*)
      SAM3_WEIGHTS="${arg#*=}"
      ;;
    OUTPUT=*)
      OUTPUT="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

echo "=============================================="
echo "Stage 2: Convert Weights"
echo "=============================================="
echo "Stage 2 weights: $STAGE2_WEIGHTS"
echo "SAM3 weights: $SAM3_WEIGHTS"
echo "Output: $OUTPUT"
echo "=============================================="

# Build command
CMD="python stage2/convert_memory_weights_stage2.py \
  --stage2_weights $STAGE2_WEIGHTS \
  --output_path $OUTPUT"

# Add SAM3 weights if specified
if [ -n "$SAM3_WEIGHTS" ]; then
  CMD="$CMD --sam3_weights $SAM3_WEIGHTS"
fi

# Run
echo "Running: $CMD"
eval $CMD

echo "Done!"
