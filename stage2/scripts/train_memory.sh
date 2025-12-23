#!/bin/bash
# --------------------------------------------------------
# Train Stage 2 Memory Modules
# --------------------------------------------------------
# Train Perceiver Resampler + Efficient Memory Attention.
#
# Usage:
#   bash stage2/scripts/train_memory.sh \
#     CFG=stage2/configs/efficient_memory.yaml \
#     OUTPUT=output/stage2_memory

set -e

# Default values
CFG="${CFG:-stage2/configs/efficient_memory.yaml}"
OUTPUT="${OUTPUT:-output/stage2_memory}"
GPUS="${GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
RESUME="${RESUME:-}"

# Parse command line arguments (KEY=VALUE format)
EXTRA_OPTS=""
for arg in "$@"; do
  case $arg in
    CFG=*)
      CFG="${arg#*=}"
      ;;
    OUTPUT=*)
      OUTPUT="${arg#*=}"
      ;;
    GPUS=*)
      GPUS="${arg#*=}"
      ;;
    MASTER_PORT=*)
      MASTER_PORT="${arg#*=}"
      ;;
    RESUME=*)
      RESUME="${arg#*=}"
      ;;
    --opts)
      # Everything after --opts goes to extra options
      shift
      EXTRA_OPTS="$@"
      break
      ;;
    *)
      # Pass through as config option
      EXTRA_OPTS="$EXTRA_OPTS $arg"
      ;;
  esac
done

echo "=============================================="
echo "Stage 2: Train Memory Modules"
echo "=============================================="
echo "Config: $CFG"
echo "Output: $OUTPUT"
echo "GPUs: $GPUS"
echo "=============================================="

# Build command
if [ "$GPUS" -gt 1 ]; then
  # Multi-GPU with torchrun
  CMD="torchrun --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    stage2/train_memory_stage2.py \
    --config $CFG \
    --output_dir $OUTPUT"
else
  # Single GPU
  CMD="python stage2/train_memory_stage2.py \
    --config $CFG \
    --output_dir $OUTPUT"
fi

# Add resume if specified
if [ -n "$RESUME" ]; then
  CMD="$CMD --resume $RESUME"
fi

# Add extra options
if [ -n "$EXTRA_OPTS" ]; then
  CMD="$CMD $EXTRA_OPTS"
fi

# Run
echo "Running: $CMD"
eval $CMD

echo "Done!"
