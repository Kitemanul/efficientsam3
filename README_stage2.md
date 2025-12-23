## Stage 2 — SAM3 Memory Bank Distillation

Stage 2 compresses the SAM3 video tracker's memory bank into efficient modules.
This stage builds on top of Stage 1 (efficient image encoders) to create a
fully efficient video object segmentation model. The approach combines
techniques from [EdgeTAM](https://arxiv.org/abs/2411.02813) and
[EfficientTAM](https://arxiv.org/abs/2502.13043) to achieve optimal
efficiency-accuracy trade-off.

### Key Innovations

1. **Perceiver Resampler** — Compresses 5184 spatial tokens (72×72) down to 512
   learnable latents (256 global + 256 2D spatial) using cross-attention.
2. **Efficient Memory Attention** — Uses 2×2 average pooling on spatial memory
   tokens with log(4) attention compensation, reducing memory complexity by 4×.
3. **Reduced Attention Layers** — Cuts memory attention from 4 layers to 2
   without significant quality loss.
4. **Pre-computed Embeddings** — Saves trunk features before FPN once, then
   reuses them for all training epochs.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EfficientSAM3 Stage 2                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │   Trunk     │ ──► │   FPN Neck      │ ──► │  Memory Encoder │   │
│  │  (Frozen)   │     │   (Frozen)      │     │    (Frozen)     │   │
│  └─────────────┘     └─────────────────┘     └─────────────────┘   │
│        │                                             │              │
│        │                                             ▼              │
│        │                              ┌───────────────────────────┐ │
│        │                              │   Perceiver Resampler     │ │
│        │                              │   (TRAINABLE, ~3M)        │ │
│        │                              │  ┌─────────────────────┐  │ │
│        │                              │  │ 256 Global Latents  │  │ │
│        │                              │  │ 256 2D Spatial      │  │ │
│        │                              │  │ 2 Encoder Layers    │  │ │
│        │                              │  └─────────────────────┘  │ │
│        │                              └───────────────────────────┘ │
│        │                                             │              │
│        ▼                                             ▼              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │          Efficient Memory Attention (TRAINABLE, ~3M)        │   │
│  │  ┌─────────────────────────────────────────────────────────┐│   │
│  │  │ • 2×2 Spatial Pooling on Memory Tokens                  ││   │
│  │  │ • log(4) Attention Compensation                         ││   │
│  │  │ • 2 Layers (reduced from 4)                             ││   │
│  │  │ • Object Pointers Kept Full Resolution                  ││   │
│  │  └─────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            │                                        │
│                            ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Mask Decoder (Frozen)                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Parameter Savings

| Component | SAM3 Teacher | EfficientSAM3 | Reduction |
|:----------|:-------------|:--------------|:----------|
| Memory Attention | 4 layers, ~12M | 2 layers, ~3M | **75% smaller** |
| Memory Tokens/Frame | 5184 (72×72) | 512 (256+256) | **90% fewer** |
| Cross-Attention FLOPs | O(N×M) | O(N×M/4) | **4× faster** |
| **Total Trainable** | - | **~6M** | - |

### Prerequisites

1. **Environment** — Follow the root [Installation](README.md#installation) guide to
   create/activate the `efficientsam3` Conda environment and run:
   ```bash
   pip install -e ".[stage2]"
   ```

2. **Stage 1 Completion** — Complete Stage 1 training first (or use provided
   Stage 1 checkpoints) as Stage 2 builds on efficient image encoders.

3. **Datasets**:
   - **SA-V** (Video): Download using `bash data/download_sa_v.sh` and extract
     frames. Data should be at `data/sa-v/extracted_frames/`.
   - **SA-1B** (10% subset for pretraining): Optional, uses static images with
     synthetic masks.

4. **Teacher Weights** — Download SAM3 tracker checkpoint from
   [Hugging Face](https://huggingface.co/facebook/sam3) or use trained Stage 1
   model as the teacher backbone.

### Directory Structure

```
stage2/
├── __init__.py                          # Module initialization
├── config.py                            # Configuration system
├── model.py                             # Teacher and student model classes
├── train_memory_stage2.py               # Main training script
├── save_video_embeddings_stage2.py      # Pre-compute trunk features
├── convert_memory_weights_stage2.py     # Merge weights with SAM3
├── configs/                             # Config files
│   └── .gitkeep
├── scripts/                             # Helper shell scripts
│   └── .gitkeep
└── data/
    ├── __init__.py
    └── sav_dataset.py                   # SA-V video dataset loader

sam3/sam3/model/
├── perceiver.py                         # PerceiverResampler implementation
└── efficient_memory_attention.py        # Efficient memory attention
```

---

## Training Pipeline

### Step 1 — Pre-compute Trunk Embeddings

**This is a one-time forward pass** through the SAM3 trunk on all video frames.
Embeddings are saved to disk and reused for all training epochs, avoiding
repeated backbone computation.

> **Note:** We save features **before the FPN** to get a single tensor shape
> `[1024, 72, 72]` per frame rather than multi-scale outputs.

```bash
# Single GPU
python stage2/save_video_embeddings_stage2.py \
  --data_dir data/sa-v/extracted_frames \
  --output_dir output/stage2_embeddings \
  --batch_size 8

# Multi-GPU (automatically distributes videos across GPUs)
torchrun --nproc_per_node=4 stage2/save_video_embeddings_stage2.py \
  --data_dir data/sa-v/extracted_frames \
  --output_dir output/stage2_embeddings \
  --batch_size 8

# 8 GPUs across 2 nodes
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
  stage2/save_video_embeddings_stage2.py \
  --data_dir data/sa-v/extracted_frames \
  --output_dir output/stage2_embeddings \
  --batch_size 8
```

The script automatically:
- Detects multi-GPU setup via `torchrun`
- Distributes videos evenly across GPUs (video `i` goes to GPU `i % num_gpus`)
- Each GPU loads the model independently and processes its assigned videos
- Saves embeddings to the same output directory (thread-safe per video)
- Synchronizes at the end to write global metadata

**Output structure**:
```
output/stage2_embeddings/
├── global_metadata.json        # Dataset-level info
├── sav_000001/
│   ├── metadata.json           # Video-level info
│   ├── 00000.pt                # [1024, 72, 72] trunk features
│   ├── 00001.pt
│   └── ...
├── sav_000002/
│   └── ...
└── ...
```

### Step 2 — Train Efficient Memory Modules

Set up your config and launch training. The training script:
- Loads pre-computed trunk embeddings (or computes on-the-fly)
- Trains Perceiver Resampler + Efficient Memory Attention
- Uses distillation loss on memory features + task losses on masks

```bash
# Basic training run
python stage2/train_memory_stage2.py \
  --config stage2/configs/efficient_memory.yaml \
  --output_dir output/stage2_memory \
  DATA.SAV_ROOT data/sa-v/extracted_frames \
  DATA.USE_PRECOMPUTED True \
  DATA.EMBEDDING_DIR output/stage2_embeddings

# Resume from checkpoint
python stage2/train_memory_stage2.py \
  --config stage2/configs/efficient_memory.yaml \
  --output_dir output/stage2_memory \
  --resume output/stage2_memory/ckpt_latest.pth
```

**Key Configuration Options**:

| Field | Description | Default |
|-------|-------------|---------|
| `DATA.SAV_ROOT` | Path to SA-V extracted frames | `data/sa-v/extracted_frames` |
| `DATA.USE_PRECOMPUTED` | Use pre-computed embeddings | `True` |
| `DATA.EMBEDDING_DIR` | Path to saved embeddings | Required if USE_PRECOMPUTED |
| `DATA.NUM_FRAMES` | Frames per training clip | `8` |
| `DATA.FRAME_SKIP` | Skip between sampled frames | `4` |
| `MODEL.PERCEIVER.NUM_LATENTS` | Global latent tokens | `256` |
| `MODEL.PERCEIVER.NUM_LATENTS_2D` | 2D spatial latent tokens | `256` |
| `MODEL.PERCEIVER.DEPTH` | Number of encoder layers | `2` |
| `MODEL.MEMORY_ATTENTION.NUM_LAYERS` | Attention layers | `2` |
| `MODEL.MEMORY_ATTENTION.POOL_SIZE` | Spatial pooling size | `2` |
| `TRAIN.LR` | Base learning rate | `3e-4` |
| `TRAIN.MAX_STEPS` | Total training steps | `130000` |
| `TRAIN.WARMUP_STEPS` | Linear warmup steps | `15000` |
| `TRAIN.BATCH_SIZE` | Batch size per GPU | `4` |

**Training settings from EdgeTAM/EfficientTAM papers**:

| Setting | Value | Source |
|---------|-------|--------|
| Optimizer | AdamW | EdgeTAM |
| LR Schedule | Cosine decay | EdgeTAM |
| Weight Decay | 0.1 | EdgeTAM |
| Gradient Clipping | 0.1 | EdgeTAM |
| Mixed Precision | FP16 | Both |
| Batch Size | 256 (effective) | EdgeTAM |

**Loss Function**:
```
L_total = 20·L_focal + 1·L_dice + 1·L_iou + 1·L_mse
```

Where:
- `L_focal` — Focal loss on predicted masks (handles class imbalance)
- `L_dice` — Dice loss for segmentation quality
- `L_iou` — IoU loss for boundary accuracy
- `L_mse` — MSE distillation loss on memory features

**Output structure**:
```
output/stage2_memory/
├── config.yaml               # Training config
├── train.log                 # Training logs
├── tensorboard/              # TensorBoard logs
├── ckpt_latest.pth           # Latest checkpoint
├── ckpt_best.pth             # Best validation checkpoint
├── ckpt_step_10000.pth       # Periodic checkpoints
└── ...
```

### Step 3 — Convert and Merge Weights

After training, merge the efficient memory modules with SAM3 (or your Stage 1
model) for end-to-end inference.

```bash
# Merge Stage 2 memory modules with SAM3
python stage2/convert_memory_weights_stage2.py \
  --stage2_weights output/stage2_memory/ckpt_best.pth \
  --sam3_weights sam3_checkpoints/sam3.pt \
  --output_path output/efficient_sam3_memory.pt

# Or merge with Stage 1 student model for full efficiency
python stage2/convert_memory_weights_stage2.py \
  --stage2_weights output/stage2_memory/ckpt_best.pth \
  --sam3_weights output/efficient_sam3_efficientvit_b0.pt \
  --output_path output/efficient_sam3_full.pt
```

**Final output structure**:
```
output/
├── stage2_embeddings/            # Pre-computed trunk features
├── stage2_memory/                # Training outputs
│   ├── ckpt_best.pth
│   └── ...
└── efficient_sam3_full.pt        # Final merged model
```

---

## Technical Details

### Perceiver Resampler (from EdgeTAM)

The Perceiver Resampler compresses spatial memory tokens from 5184 (72×72) to
512 learnable latents:

- **Global Latents (256)**: Attend to all spatial positions globally
- **2D Spatial Latents (256)**: Arranged in 16×16 grid, use window attention

Each encoder layer:
1. Cross-attention: Latents query the input features
2. Self-attention: Latents attend to each other (optional)
3. FFN: Standard feed-forward network

```python
# From sam3/sam3/model/perceiver.py
class EfficientSpatialPerceiver(nn.Module):
    """
    Combines global and 2D spatial latents for memory compression.
    Global latents attend globally, 2D latents use window attention.
    """
```

### Efficient Memory Attention (from EfficientTAM)

The memory attention uses 2×2 average pooling on spatial memory tokens:

```python
# Key insight: Pool spatial dimensions before cross-attention
# Reduces memory tokens from 5184 → 1296 (4× reduction)

# Compensation: Scale attention by log(pool_size²) = log(4) ≈ 1.39
# This accounts for the reduced variance after pooling
```

Object pointer tokens are kept at full resolution since they encode
instance-specific information.

### Training Strategy

1. **Freeze all SAM3 components** — Only train Perceiver + Memory Attention
2. **Pre-compute embeddings** — Avoid loading full SAM3 every iteration
3. **Video-based training** — Sample 8-frame clips with skip=4
4. **Distillation** — MSE loss on memory features vs teacher

---

## Model Zoo

| Model | Backbone | Memory | Params | Download |
|:------|:---------|:-------|:-------|:---------|
| ES3-RV-M-Mem | RepViT-M1.1 | Efficient | ~13M | Coming soon |
| ES3-TV-M-Mem | TinyViT-11M | Efficient | ~16M | Coming soon |
| ES3-EV-M-Mem | EfficientViT-B1 | Efficient | ~10M | Coming soon |

---

## Benchmarks

### SA-V Validation Set

| Model | J&F Mean | J Mean | F Mean | FPS |
|:------|:---------|:-------|:-------|:----|
| SAM3 (Teacher) | - | - | - | - |
| EfficientSAM3 | - | - | - | - |

### DAVIS 2017 Val

| Model | J&F Mean | J Mean | F Mean |
|:------|:---------|:-------|:-------|
| SAM3 (Teacher) | - | - | - |
| EfficientSAM3 | - | - | - |

*Benchmarks to be filled after training completion*

---

## Citation

If you use Stage 2 of EfficientSAM3, please cite:

```bibtex
@article{efficientsam3,
  title={EfficientSAM3: Efficient Segment Anything Model 3},
  author={Your Name},
  year={2024}
}

@article{edgetam,
  title={EdgeTAM: On-Device Track Anything Model},
  author={Zhou, Chong and Li, Chenchen and Rountree, Blake and Shen, Xide and Wang, Jialiang and Qi, Haotian},
  journal={arXiv preprint arXiv:2411.02813},
  year={2024}
}

@article{efficienttam,
  title={EfficientTAM: Efficient Track Anything Model},
  author={Zhang, Zhiyuan and Zou, Yang},
  journal={arXiv preprint arXiv:2502.13043},
  year={2025}
}
```

---

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `DATA.BATCH_SIZE` (try 2 or 1)
2. Reduce `DATA.NUM_FRAMES` (try 4)
3. Enable gradient checkpointing (add to config)
4. Use pre-computed embeddings (`DATA.USE_PRECOMPUTED=True`)

### Slow Training

1. Use pre-computed embeddings to skip backbone forward
2. Increase `DATA.NUM_WORKERS` (try 8 or 16)
3. Enable mixed precision (`TRAIN.AMP=True`)

### Poor Convergence

1. Check learning rate (default 3e-4 works well)
2. Increase warmup steps (try 20K)
3. Verify teacher embeddings are correct
4. Check data augmentation settings

### Import Errors

```bash
# Make sure sam3 is in path
export PYTHONPATH=$PYTHONPATH:/path/to/efficientsam3/sam3

# Or install in development mode
pip install -e .
```
