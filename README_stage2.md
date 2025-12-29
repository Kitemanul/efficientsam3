## Stage 2 — Efficient Memory Training

Stage 2 focuses on training the efficient memory modules for the SAM3 video tracker.
We adopt the direct training strategy from [EfficientTAM](https://arxiv.org/abs/2502.13043) and [SAM 2](https://arxiv.org/abs/2408.00714).
We initialize the model with a pre-trained SAM3 image encoder (frozen) and train
only the lightweight memory components on the SA-V dataset.

### Key Innovations

1. **Perceiver Resampler** — Compresses dense spatial memory features into a small set of
   learnable latents (e.g., 128 tokens), significantly reducing memory storage and computation.
2. **Efficient Memory Attention** — Utilizes `EfficientRoPEAttention` from EfficientTAM,
   which applies spatial pooling to memory keys/values for faster cross-attention.
3. **Efficient Memory Encoder** — A lightweight encoder that fuses image features and masks
   into the memory bank.

### Architecture Overview

The `EfficientSAM3Stage2` model is built upon the `SAM2Base` architecture but replaces
critical memory components with efficient variants:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EfficientSAM3 Stage 2                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────────┐                            │
│  │ SAM3 Trunk  │ ──► │ SAM3 Neck       │ ──► [Image Features]       │
│  │  (Frozen)   │     │   (Frozen)      │             │              │
│  └─────────────┘     └─────────────────┘             │              │
│                                                      ▼              │
│                                       ┌───────────────────────────┐ │
│                                       │      Memory Encoder       │ │
│                                       │    (EfficientTAM Style)   │ │
│                                       └──────────────┬────────────┘ │
│                                                      │              │
│                                                      ▼              │
│                                       ┌───────────────────────────┐ │
│                                       │    Perceiver Resampler    │ │
│                                       │   (Compresses to Latents) │ │
│                                       └──────────────┬────────────┘ │
│                                                      │              │
│        ┌─────────────────────────────────────────────▼──────────┐   │
│        │          Efficient Memory Attention                    │   │
│        │      (Queries: Current Frame, Keys/Vals: Memory)       │   │
│        └─────────────────────────────┬──────────────────────────┘   │
│                                      │                              │
│                                      ▼                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Mask Decoder (SAM2/SAM3)                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Prerequisites

1. **Environment**: Ensure `sam2`, `sam3`, and `EfficientTAM` dependencies are installed.
2. **Weights**: Place the SAM3 checkpoint at `sam3_checkpoints/sam3.pt`.
3. **Data**: Prepare the SA-V dataset.

### Directory Structure

```
stage2/
├── model.py                             # EfficientSAM3Stage2 model definition
├── train_memory_stage2.py               # Main training script
├── convert_memory_weights_stage2.py     # Script to merge text encoder back
└── data/                                # Data loaders
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
# Single GPU
bash stage2/scripts/train_memory.sh \
  GPUS=1 \
  BATCH_SIZE=4 \
  DATA_PATH=data/sa-v/extracted_frames

# Multi-GPU (e.g., 8 GPUs)
bash stage2/scripts/train_memory.sh \
  GPUS=8 \
  BATCH_SIZE=4 \
  DATA_PATH=data/sa-v/extracted_frames
```

**Manual Execution:**

**Single GPU Training:**
```bash
python stage2/train_memory_stage2.py \
  --sam3_checkpoint sam3_checkpoints/sam3.pt \
  --data_path data/sa-v/extracted_frames \
  --output_dir output/stage2_checkpoints \
  --batch_size 4
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
