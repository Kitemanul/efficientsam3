# --------------------------------------------------------
# Stage 2 Memory Distillation Configuration
# --------------------------------------------------------

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 4  # Videos per batch (each has multiple frames)
# Preferred naming (matches README_stage2.md + stage2/configs/*.yaml)
_C.DATA.SAV_ROOT = 'data/sa-v/extracted_frames'  # Path to SA-V extracted_frames
# Backwards-compat alias (deprecated)
_C.DATA.DATA_PATH = _C.DATA.SAV_ROOT
_C.DATA.DATASET = 'sav'  # 'sav' for SA-V dataset
_C.DATA.MEAN = [123.675, 116.28, 103.53]
_C.DATA.STD = [58.395, 57.12, 57.375]
_C.DATA.IMG_SIZE = 1008  # SAM3 native resolution
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 8
_C.DATA.DEBUG = False

# Video-specific settings
_C.DATA.NUM_FRAMES = 8  # Number of frames per video clip (preferred)
_C.DATA.FRAMES_PER_VIDEO = _C.DATA.NUM_FRAMES  # Backwards-compat alias
_C.DATA.MAX_OBJECTS = 3  # Maximum objects per video
_C.DATA.FRAME_SKIP = 4  # Sample every N frames (for temporal diversity)
_C.DATA.TRAIN_SPLIT = 0.95
_C.DATA.VAL_SPLIT = 0.05

# Pre-computed embeddings
_C.DATA.USE_PRECOMPUTED = True  # Whether to use pre-computed embeddings
_C.DATA.EMBEDDING_DIR = 'output/stage2_embeddings'  # Preferred name
_C.DATA.EMBED_PATH = _C.DATA.EMBEDDING_DIR  # Backwards-compat alias

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'memory_trainer'
_C.MODEL.NAME = 'efficient_sam3_memory'
_C.MODEL.RESUME = ''  # SAM3 checkpoint path
_C.MODEL.PRETRAINED = ''  # Pretrained memory module weights

# Perceiver settings (from EdgeTAM)
_C.MODEL.PERCEIVER = CN()
_C.MODEL.PERCEIVER.ENABLED = True
_C.MODEL.PERCEIVER.NUM_LATENTS = 256  # Global latents
_C.MODEL.PERCEIVER.NUM_LATENTS_2D = 256  # 2D spatial latents (16x16 grid)
_C.MODEL.PERCEIVER.DEPTH = 2  # Number of encoder layers
_C.MODEL.PERCEIVER.HEADS = 8  # Number of attention heads
_C.MODEL.PERCEIVER.DIM_HEAD = 64  # Dimension per head
_C.MODEL.PERCEIVER.USE_SELF_ATTN = True  # Use self-attention in layers

# Memory attention settings (from EfficientTAM + EdgeTAM)
_C.MODEL.MEMORY_ATTENTION = CN()
_C.MODEL.MEMORY_ATTENTION.NUM_LAYERS = 2  # Reduced from 4
_C.MODEL.MEMORY_ATTENTION.POOL_SIZE = 2  # 2x2 pooling
_C.MODEL.MEMORY_ATTENTION.D_MODEL = 256
_C.MODEL.MEMORY_ATTENTION.DIM_FEEDFORWARD = 2048
_C.MODEL.MEMORY_ATTENTION.NUM_HEADS = 8
_C.MODEL.MEMORY_ATTENTION.DROPOUT = 0.0

# Memory encoder settings
_C.MODEL.MEMORY_ENCODER = CN()
_C.MODEL.MEMORY_ENCODER.OUT_DIM = 64
_C.MODEL.MEMORY_ENCODER.IN_DIM = 256

# Tracker settings
_C.MODEL.TRACKER = CN()
_C.MODEL.TRACKER.NUM_MASKMEM = 7  # Number of memory frames
_C.MODEL.TRACKER.IMAGE_SIZE = 1008
_C.MODEL.TRACKER.BACKBONE_STRIDE = 14

# -----------------------------------------------------------------------------
# Distillation settings
# -----------------------------------------------------------------------------
_C.DISTILL = CN()
_C.DISTILL.ENABLED = True
_C.DISTILL.TEACHER_EMBED_PATH = ''  # Path to teacher embeddings

# Feature distillation (like EdgeTAM)
_C.DISTILL.FEATURE_LOSS = True  # MSE on memory-conditioned features
_C.DISTILL.FEATURE_LOSS_WEIGHT = 1.0

# Mask distillation
_C.DISTILL.MASK_LOSS = True  # Task loss on mask predictions
_C.DISTILL.FOCAL_WEIGHT = 20.0
_C.DISTILL.DICE_WEIGHT = 1.0
_C.DISTILL.IOU_WEIGHT = 1.0
_C.DISTILL.OCCLUSION_WEIGHT = 1.0
_C.DISTILL.MSE_WEIGHT = 1.0
_C.DISTILL.FOCAL_ALPHA = 0.25
_C.DISTILL.FOCAL_GAMMA = 2.0

# Embedding dimensions
_C.DISTILL.EMBED_DIM = 1024  # Trunk feature dimension
_C.DISTILL.EMBED_SIZE = 72  # Feature map spatial size

# -----------------------------------------------------------------------------
# Training settings (from EdgeTAM/EfficientTAM)
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 50
_C.TRAIN.STEPS = 130000  # Total training steps (legacy name)
_C.TRAIN.MAX_STEPS = _C.TRAIN.STEPS  # Preferred name (matches README)
_C.TRAIN.WARMUP_STEPS = 15000

# Learning rates (EdgeTAM style: lower for frozen components)
_C.TRAIN.BASE_LR = 3e-4  # Legacy name
_C.TRAIN.LR = _C.TRAIN.BASE_LR  # Preferred name
_C.TRAIN.BACKBONE_LR = 0.0  # Frozen
_C.TRAIN.WARMUP_LR = 1e-7
_C.TRAIN.MIN_LR = 1e-5

# Optimizer
_C.TRAIN.WEIGHT_DECAY = 0.1
_C.TRAIN.CLIP_GRAD = 0.1  # Legacy name
_C.TRAIN.GRAD_CLIP = _C.TRAIN.CLIP_GRAD  # Preferred name
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.AMP = True
_C.TRAIN.LOG_INTERVAL = 100
_C.TRAIN.VAL_INTERVAL = 5000
_C.TRAIN.SAVE_INTERVAL = 10000

# Scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# Progressive training (EdgeTAM)
_C.TRAIN.PROGRESSIVE = CN()
_C.TRAIN.PROGRESSIVE.ENABLED = False
_C.TRAIN.PROGRESSIVE.STAGE1_FRAMES = 8  # Initial frame count
_C.TRAIN.PROGRESSIVE.STAGE2_FRAMES = 16  # After warmup
_C.TRAIN.PROGRESSIVE.STAGE3_FRAMES = 32  # Final stage

# -----------------------------------------------------------------------------
# Augmentation settings (from EdgeTAM)
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.HFLIP = True
_C.AUG.AFFINE = True
_C.AUG.AFFINE_DEGREE = 25
_C.AUG.AFFINE_SHEAR = 20
_C.AUG.COLOR_JITTER = True
_C.AUG.COLOR_JITTER_BRIGHTNESS = 0.1
_C.AUG.COLOR_JITTER_CONTRAST = 0.03
_C.AUG.COLOR_JITTER_SATURATION = 0.03
_C.AUG.GRAYSCALE = True
_C.AUG.GRAYSCALE_PROB = 0.05
_C.AUG.RANDOM_FLIP = True  # Alias used in stage2/configs/efficient_memory.yaml
_C.AUG.COLOR_JITTER = 0.4  # Alias used in stage2/configs/efficient_memory.yaml

# -----------------------------------------------------------------------------
# Evaluation settings
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.INTERVAL = 1000  # Eval every N steps
_C.EVAL.DATASETS = ['davis', 'mose']  # Evaluation datasets
_C.EVAL.ENABLED = False
_C.EVAL.EVAL_INTERVAL = 10000

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.AMP_ENABLE = True  # Legacy top-level AMP flag (prefer TRAIN.AMP)
_C.OUTPUT = ''
_C.TAG = 'default'
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 10
_C.SEED = 42
_C.EVAL_MODE = False
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
            config.defrost()
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    if hasattr(args, 'batch_size') and args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if hasattr(args, 'data_path') and args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if hasattr(args, 'pretrained') and args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if hasattr(args, 'resume') and args.resume:
        config.MODEL.RESUME = args.resume
    if hasattr(args, 'accumulation_steps') and args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if hasattr(args, 'disable_amp') and args.disable_amp:
        config.AMP_ENABLE = False
    if hasattr(args, 'output') and args.output:
        config.OUTPUT = args.output
    if hasattr(args, 'tag') and args.tag:
        config.TAG = args.tag
    if hasattr(args, 'eval') and args.eval:
        config.EVAL_MODE = True

    # Compute embedding save path if not set
    if not config.DISTILL.TEACHER_EMBED_PATH:
        config.DISTILL.TEACHER_EMBED_PATH = os.path.join(
            config.OUTPUT, 'stage2_teacher', 'embeddings'
        )

    config.freeze()


def get_config(args=None):
    """Get config from args or return default config."""
    config = _C.clone()
    if args is not None:
        update_config(config, args)
    return config


def update_config_from_file(config, cfg_file):
    """Update config from a YAML file."""
    _update_config_from_file(config, cfg_file)
    return config
