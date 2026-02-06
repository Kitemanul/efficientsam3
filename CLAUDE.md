# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EfficientSAM3 distills SAM3 (Segment Anything Model 3) into lightweight, edge-deployable models via three-stage progressive knowledge distillation. Currently Stage 1 (encoder distillation) is implemented; Stages 2-3 are planned.

## Setup & Installation

```bash
conda create -n efficientsam3 python=3.12 -y && conda activate efficientsam3
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[stage1]"
```

Dev tools: `pip install -e ".[dev]"` (pytest, black==24.2.0, ruff, mypy)

## Common Commands

```bash
# Stage 1: Save teacher embeddings then train
python stage1/save_embedding_image_stage1.py --cfg stage1/configs/teacher/sam_vit_huge_sa1b.yaml
python stage1/train_image_encoder_stage1.py --cfg stage1/configs/es_rv_m.yaml

# Text encoder training
python stage1/save_embedding_text_stage1.py --cfg stage1/configs/teacher/sam_text_teacher.yaml
python stage1/train_text_encoder_stage1.py --cfg stage1/configs/es_mc_s.yaml

# Weight conversion (splice student encoder back into full SAM3 checkpoint)
python stage1/convert_image_encoder_weights_stage1.py
python stage1/convert_text_encoder_weights_stage1.py
python stage1/convert_both_encoders_weights_stage1.py

# Evaluation
python eval/eval_coco.py --coco_root data/coco --output_dir output
python eval/eval_text_encoder_similarity.py --student-ckpt path/to/ckpt.pth --np-json data/sa-v-text/sa-co-veval/saco_veval_noun_phrases.json

# ONNX export
python export_onnx.py          # Full pipeline (all 10 components)
python export_image_model_onnx.py  # Image model only

# Formatting (matches CI)
black --check sam3/ stage1/
ruff check sam3/ stage1/
```

## Architecture

### Directory Layout

- `sam3/` — Vendored SAM3 package (the teacher model and shared components)
  - `sam3/model/` — Core model: encoder, decoder, memory, geometry encoders, text encoder
  - `sam3/backbones/` — Student backbone implementations (RepViT, TinyViT, EfficientViT, MobileCLIP)
  - `sam3/model_builder.py` — Model construction functions (`build_sam3_image_model`, `build_efficientsam3_image_model`)
  - `sam3/train/` — Training infrastructure (trainer, matcher, losses)
- `stage1/` — Stage 1 encoder distillation pipeline
  - `model.py` — Student/teacher encoder builders and adapter classes
  - `config.py` — YACS-based configuration system
  - `configs/` — YAML configs (base + per-backbone overrides)
  - `data/` — Custom data loaders for SA-1B and Recap-DataComp-1B
- `eval/` — Evaluation scripts (COCO mIoU, text encoder cosine similarity)
- `data/` — Dataset download scripts (`download_*.sh`)
- `export_onnx.py` — ONNX export with wrapper classes for all 10 exportable components

### Model Architecture

**Sam3Image** is the core model. The student replaces only the vision/text backbone while keeping all other components (transformer encoder/fusion, decoder, geometry encoders, segmentation head) from the original SAM3.

Key entry points:
- `sam3/model_builder.py:build_sam3_image_model()` — Builds teacher SAM3
- `sam3/model_builder.py:build_efficientsam3_image_model()` — Builds student EfficientSAM3
- `stage1/model.py:build_image_student_model()` — Builds standalone student encoder for distillation
- `stage1/model.py:build_efficient_sam3()` — Splices student encoder into full SAM3 model

### Student Backbone Adapters (stage1/model.py)

Each backbone family has an adapter that normalizes the output to `[B, C, H, W]` format:
- `RepViTAdapter` — Wraps RepViT M0.9/M1.1/M2.3
- `TinyViTAdapter` — Wraps TinyViT 5M/11M/21M
- `EfficientViTAdapter` — Wraps EfficientViT B0/B1/B2

The `ImageStudentEncoder` adds a projection head (Conv2d→BN→GELU→Conv2d) and interpolates to match teacher embed size (72×72).

### ONNX Export Components

**Exportable (10 total):**
- Detector: Image Encoder, Text Encoder, DotProductScoring, Box Head
- Tracker: Prompt Encoder, Mask Decoder, Memory Encoder, obj_ptr_proj, obj_ptr_tpos_proj, mask_downsample

**Not exportable (run in Python):** Memory Attention, Geometry Encoder, Transformer Fusion

### Configuration System

YACS-based. Base config: `stage1/configs/base_stage1.yaml`. Model-specific overrides named `es_{rv,tv,ev}_{s,m,l}.yaml` for image encoders and `es_mc_{s,s1,l}.yaml` for text encoders. Key config sections: `MODEL`, `TRAIN`, `DATA`, `DISTILL`.

### Weight Prefix Convention

Checkpoint keys use `tracker.*` and `detector.*` prefixes. When converting student weights back into a full SAM3 checkpoint, prefix mapping must match exactly.

### Distillation Losses

Configured in `DISTILL` section: pixel-wise MSE (weight 1.0), cosine similarity (weight 1.0), channel-wise correlation (weight 0.0 by default).

## Communication

使用中文回答，简洁直接。
