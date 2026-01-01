## Stage 2 — Efficient Memory Training

Stage 2 focuses on training the efficient memory modules for the SAM3 video tracker.
We adopt the direct training strategy from [EfficientTAM](https://arxiv.org/abs/2502.13043) and [SAM 2](https://arxiv.org/abs/2408.00714).
We initialize the model with SAM3's frozen vision backbone and SAM heads, then train
only the lightweight memory components on the SA-V dataset.

### Key Innovations

1. **Perceiver Resampler** — Compresses dense spatial memory features (5184 tokens) into 64
   learnable latents, achieving 81x compression while preserving tracking quality.
2. **Efficient Memory Attention** — Lightweight cross-attention with optional spatial pooling
   for faster memory fusion.
3. **Efficient Memory Encoder** — `SimpleMaskEncoder` that fuses image features and masks
   into compact memory representations.

### Architecture Overview

The `EfficientSam3Train` model extends `Sam3TrackerBase` with efficient memory components.
The code is **fully self-contained** within `sam3/` and `stage2/` folders.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 2 ARCHITECTURE (~470M params)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────┐                                             │
│  │   SAM3 Vision Backbone     │  ❄️ FROZEN (462M)                           │
│  │  (Sam3DualViTDetNeck)      │  Multi-scale image features                │
│  └─────────────┬──────────────┘                                             │
│                │ backbone_fpn: [level0, level1, level2]                     │
│                ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      PER-FRAME TRACKING LOOP                         │   │
│  │  Frame 0: GT mask → SimpleMaskEncoder → Memory Bank                  │   │
│  │  Frame 1..T:                                                         │   │
│  │    Memory → PerceiverResampler (compress) → EfficientMemoryAttention │   │
│  │    → SAM Mask Decoder → Prediction → Update Memory                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  TRAINABLE MODULES (3.8M):                     FROZEN MODULES:              │
│  ┌──────────────────────┐                      ┌──────────────────────┐    │
│  │ SimpleMaskEncoder    │ 1.4M 🔥              │ Vision Backbone      │    │
│  │ PerceiverResampler   │ 0.2M 🔥              │ SAM Mask Decoder     │    │
│  │ EfficientMemoryAttn  │ 2.1M 🔥              │ SAM Prompt Encoder   │    │
│  │ mem_proj, mem_pos    │ 0.1M 🔥              │ obj_ptr_proj, etc.   │    │
│  └──────────────────────┘                      └──────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Prerequisites

1. **Environment**: Install the `sam3` package from this repo (`pip install -e .` in `sam3/`)
2. **Weights**: Place the SAM3 checkpoint at `sam3_checkpoints/sam3.pt`
3. **Data**: Download and prepare the SA-V dataset


### Directory Structure

```
stage2/
├── __init__.py                          # Re-exports EfficientSam3Train, etc.
├── train_memory_stage2.py               # Main training script
├── verify_training_pipeline.py          # Verification script
├── data/
│   └── sav_dataset.py                   # SA-V video dataset loader
└── scripts/
    └── train_memory.sh                  # Training launch script

sam3/sam3/model/
├── efficient_sam3_train.py              # EfficientSam3Train model (training)
├── efficient_sam3_tracker.py            # EfficientSam3TrackerBase (inference)
├── efficient_sam3_model_builder.py      # Model builder functions
├── efficient_memory_attention.py        # EfficientMemoryAttention module
├── perceiver.py                         # PerceiverResampler module
└── memory.py                            # SimpleMaskEncoder, etc.
```

### Training Pipeline

#### Step 1 — Train Efficient Memory Modules

Set up your config and launch training. The training script:
- Loads the SAM3 model (Vision + Prompt + Mask Decoder)
- Freezes the backbone and other non-memory components
- Trains Perceiver Resampler + Efficient Memory Attention + Memory Encoder
- Uses Dice Loss and Focal Loss on predicted masks

**Using the Helper Script (Recommended):**
We provide a bash script `stage2/scripts/train_memory.sh` that handles multi-GPU setup via `torchrun`.

```bash
# Single GPU - Full dataset
bash stage2/scripts/train_memory.sh \
  GPUS=1 \
  BATCH_SIZE=4 \
  DATA_PATH=data/sa-v/formatted

# Single GPU - 1% subset (for quick experiments)
bash stage2/scripts/train_memory.sh \
  GPUS=1 \
  BATCH_SIZE=4 \
  DATA_PATH=data/sa-v/formatted \
  SUBSET_FRACTION=0.01

# Multi-GPU (e.g., 8 GPUs)
bash stage2/scripts/train_memory.sh \
  GPUS=8 \
  BATCH_SIZE=4 \
  DATA_PATH=data/sa-v/formatted
```

**Subset Training:**

Use `SUBSET_FRACTION` to train on a percentage of video folders (useful for debugging or quick experiments):

| Fraction | Description | Use Case |
|----------|-------------|----------|
| `0.01` | 1% of videos | Quick debugging, smoke tests |
| `0.10` | 10% of videos | Hyperparameter tuning |
| `0.50` | 50% of videos | Ablation studies |
| `1.0` | All videos (default) | Full training |

*Note: At least 1 video folder is always used, even if the calculated subset would be 0.*

**Manual Execution:**

**Single GPU Training:**
```bash
python stage2/train_memory_stage2.py \
  --sam3_checkpoint sam3_checkpoints/sam3.pt \
  --data_path data/sa-v/extracted_frames \
  --output_dir output/stage2_checkpoints \
  --batch_size 4 \
  --subset_fraction 0.01  # Use 1% of data
```

**Multi-GPU Training (e.g., 8 GPUs):**
```bash
torchrun --nproc_per_node=8 stage2/train_memory_stage2.py \
  --sam3_checkpoint sam3_checkpoints/sam3.pt \
  --data_path data/sa-v/extracted_frames \
  --output_dir output/stage2_checkpoints \
  --batch_size 4
```

*Note: The current script uses a dummy dataset for verification if `data_path` is not found. Ensure you have downloaded and extracted the SA-V dataset to the specified path.*

#### Step 2 — Merge Weights

After training, merge the trained memory weights with the original SAM3 text encoder to create a full inference-ready checkpoint.

```bash
python stage2/convert_memory_weights_stage2.py \
    --trained_checkpoint output/stage2_checkpoints/model_final.pth \
    --sam3_checkpoint sam3_checkpoints/sam3.pt \
    --output_path output/efficient_sam3_stage2_full.pth
```

The resulting `efficient_sam3_stage2_full.pth` can be loaded directly by the SAM3 inference pipeline.
