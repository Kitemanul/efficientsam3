# --------------------------------------------------------
# SA-V Video Dataset for Stage 2 Training
# --------------------------------------------------------

"""
SA-V Dataset loader for video object segmentation training.

The SA-V dataset contains video frames in the following structure:
    extracted_frames/
        sav_000001/
            00000.jpg
            00001.jpg
            ...
            sav_000001_auto.json    # Automatic annotations
            sav_000001_manual.json  # Manual annotations

Each video has frames at 24 FPS, with annotations at 6 FPS.
"""

import os
import json
import random
from typing import Optional, List, Dict, Tuple, Any

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_image
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils


class SAVVideoDataset(Dataset):
    """
    SA-V Video Dataset for memory bank training.
    
    Loads video clips with multiple frames for training the memory
    attention module. Supports both raw images and pre-computed embeddings.
    
    Args:
        data_path: Path to extracted_frames directory
        frames_per_video: Number of frames to sample per video clip
        frame_skip: Sample every N frames (for temporal diversity)
        img_size: Target image size (default 1008 for SAM3)
        max_objects: Maximum number of objects to track per video
        transform: Optional transform for augmentation
        use_precomputed: Whether to use pre-computed embeddings
        embed_path: Path to pre-computed embeddings
        split: 'train' or 'val'
        subset_fraction: Fraction of videos to use (0.0-1.0). Default 1.0 uses all.
                        Use 0.01 for 1% of data for quick experiments.
    """
    
    def __init__(
        self,
        data_path: str,
        frames_per_video: int = 8,
        frame_skip: int = 4,
        img_size: int = 1008,
        max_objects: int = 3,
        transform: Optional[transforms.Compose] = None,
        use_precomputed: bool = False,
        embed_path: str = '',
        split: str = 'train',
        mean: List[float] = [123.675, 116.28, 103.53],
        std: List[float] = [58.395, 57.12, 57.375],
        subset_fraction: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.frames_per_video = frames_per_video
        self.frame_skip = frame_skip
        self.img_size = img_size
        self.max_objects = max_objects
        self.transform = transform
        self.use_precomputed = use_precomputed
        self.embed_path = embed_path
        self.split = split
        self.subset_fraction = max(0.0, min(1.0, subset_fraction))  # Clamp to [0, 1]
        self.seed = seed
        
        # Normalization values
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        
        # Discover videos (with optional subset)
        self.videos = self._discover_videos()
        
        # Apply subset fraction if less than 1.0
        if self.subset_fraction < 1.0 and len(self.videos) > 0:
            random.seed(self.seed)  # Reproducible subset
            num_videos = max(1, int(len(self.videos) * self.subset_fraction))
            self.videos = random.sample(self.videos, num_videos)
            self.videos = sorted(self.videos)  # Keep sorted order
            print(f"Using {self.subset_fraction*100:.1f}% subset: {len(self.videos)} videos")
        else:
            print(f"Found {len(self.videos)} videos in {data_path}")
        
        # Load or create annotations index
        self.annotations = self._load_annotations()

    def _discover_videos(self) -> List[str]:
        """Discover all video directories."""
        videos = []
        if not os.path.exists(self.data_path):
            print(f"Warning: Data path {self.data_path} does not exist")
            return videos
            
        for name in sorted(os.listdir(self.data_path)):
            video_dir = os.path.join(self.data_path, name)
            if os.path.isdir(video_dir):
                # Check if it has frames
                frames = [f for f in os.listdir(video_dir) if f.endswith('.jpg')]
                if len(frames) >= self.frames_per_video:
                    videos.append(name)
        return videos

    def _load_annotations(self) -> Dict[str, Dict]:
        """Load mask annotations for each video."""
        annotations = {}
        for video_name in self.videos:
            video_dir = os.path.join(self.data_path, video_name)
            
            # Try to load manual annotations first, then auto
            manual_path = os.path.join(video_dir, f'{video_name}_manual.json')
            auto_path = os.path.join(video_dir, f'{video_name}_auto.json')
            
            if os.path.exists(manual_path):
                with open(manual_path, 'r') as f:
                    annotations[video_name] = json.load(f)
            elif os.path.exists(auto_path):
                with open(auto_path, 'r') as f:
                    annotations[video_name] = json.load(f)
            else:
                # No annotations - use dummy for training
                annotations[video_name] = {'masklet': []}
                
        return annotations

    def _get_frame_paths(self, video_name: str) -> List[str]:
        """Get sorted list of frame paths for a video."""
        video_dir = os.path.join(self.data_path, video_name)
        frames = [
            f for f in os.listdir(video_dir) 
            if f.endswith('.jpg') or f.endswith('.png')
        ]
        frames = sorted(frames)
        return [os.path.join(video_dir, f) for f in frames]

    def _sample_frame_indices(self, num_frames: int) -> List[int]:
        """
        Sample frame indices for a video clip.
        
        Uses frame_skip to get temporal diversity while staying
        within the video bounds.
        """
        max_start = num_frames - (self.frames_per_video * self.frame_skip)
        
        if max_start <= 0:
            # Video too short - sample uniformly
            indices = np.linspace(0, num_frames - 1, self.frames_per_video)
            return [int(i) for i in indices]
        
        start = random.randint(0, max_start)
        indices = [start + i * self.frame_skip for i in range(self.frames_per_video)]
        return indices

    def _load_frame(self, frame_path: str) -> torch.Tensor:
        """Load and preprocess a single frame."""
        # Load image
        img = Image.open(frame_path).convert('RGB')
        
        # Resize to target size
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Convert to tensor [C, H, W] in range [0, 255]
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        
        # Normalize
        img = (img - self.mean) / self.std
        
        return img

    def _get_mask_for_frame(
        self,
        video_name: str,
        frame_idx: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Get mask annotation for a specific frame.
        
        SA-V format: masklet[frame_idx] is a list of RLE dicts for that frame.
        Each RLE dict has 'size' and 'counts' keys.
        
        Returns a tensor of shape [num_objects, H, W] with binary masks.
        """
        annotations = self.annotations.get(video_name, {})
        masklets = annotations.get('masklet', [])
        
        # Get original video dimensions for proper mask decoding
        orig_height = int(annotations.get('video_height', height))
        orig_width = int(annotations.get('video_width', width))
        
        masks = []
        
        # SA-V annotations are at 6 FPS, video is at 24 FPS
        # Map frame index to annotation frame index
        anno_frame_idx = frame_idx // 4
        
        if anno_frame_idx < len(masklets):
            # masklets[anno_frame_idx] is a list of RLE masks for this frame
            frame_masks = masklets[anno_frame_idx]
            
            for obj_idx in range(min(len(frame_masks), self.max_objects)):
                rle = frame_masks[obj_idx]
                if rle and isinstance(rle, dict) and 'counts' in rle:
                    mask = self._decode_rle(rle, orig_height, orig_width)
                    # Resize mask to target size
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=(height, width),
                        mode='nearest'
                    ).squeeze(0).squeeze(0)
                else:
                    mask = torch.zeros(height, width)
                masks.append(mask)
        
        # Pad to max_objects
        while len(masks) < self.max_objects:
            masks.append(torch.zeros(height, width))
        
        return torch.stack(masks, dim=0)  # [num_objects, H, W]

    def _decode_rle(
        self,
        rle: Any,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Decode RLE mask to binary mask."""
        # SA-V RLE is typically COCO format
        if isinstance(rle, str):
            # Compressed RLE string
            rle_obj = {'counts': rle.encode('utf-8'), 'size': [height, width]}
        elif isinstance(rle, dict):
            rle_obj = rle
        else:
            # Assume it's uncompressed counts or similar
            rle_obj = {'counts': rle, 'size': [height, width]}
            
        mask = mask_utils.decode(rle_obj)
        return torch.from_numpy(mask).float()

    def _generate_point_prompt(
        self,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate point prompts from a mask.
        
        Returns:
            coords: [N, 2] - point coordinates (x, y)
            labels: [N] - point labels (1 for positive, 0 for negative)
        """
        # Find positive points (inside mask)
        if mask.sum() > 0:
            ys, xs = torch.where(mask > 0.5)
            idx = random.randint(0, len(xs) - 1)
            pos_point = torch.tensor([[xs[idx].item(), ys[idx].item()]])
            pos_label = torch.tensor([1])
        else:
            # No mask - random point with negative label
            pos_point = torch.tensor([[mask.shape[1] // 2, mask.shape[0] // 2]])
            pos_label = torch.tensor([0])
        
        return pos_point, pos_label

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a video clip for training.
        
        Returns:
            dict with:
                - frames: [T, C, H, W] - video frames
                - masks: [T, num_objects, H, W] - mask annotations
                - point_coords: [num_objects, 1, 2] - point prompts
                - point_labels: [num_objects, 1] - point labels
                - video_name: str
                - frame_indices: List[int]
        """
        video_name = self.videos[idx]
        frame_paths = self._get_frame_paths(video_name)
        
        # Sample frame indices
        frame_indices = self._sample_frame_indices(len(frame_paths))
        
        # Load frames
        frames = []
        for frame_idx in frame_indices:
            frame = self._load_frame(frame_paths[frame_idx])
            frames.append(frame)
        frames = torch.stack(frames, dim=0)  # [T, C, H, W]
        
        # Get masks for all frames (for teacher forcing)
        mask_size = self.img_size // 4  # Low-res mask size
        all_masks = []
        for frame_idx in frame_indices:
            mask = self._get_mask_for_frame(
                video_name, frame_idx, mask_size, mask_size
            )
            all_masks.append(mask)
        all_masks = torch.stack(all_masks, dim=0)  # [T, num_objects, H, W]
        
        # Generate point prompts from first frame masks
        point_coords = []
        point_labels = []
        for obj_idx in range(self.max_objects):
            coords, labels = self._generate_point_prompt(all_masks[0, obj_idx])
            point_coords.append(coords)
            point_labels.append(labels)
        point_coords = torch.stack(point_coords, dim=0)  # [num_objects, 1, 2]
        point_labels = torch.stack(point_labels, dim=0)  # [num_objects, 1]
        
        # Apply augmentation if provided
        if self.transform is not None:
            # Note: need to apply same transform to all frames
            frames = self.transform(frames)
        
        return {
            'frames': frames,  # [T, C, H, W]
            'masks': all_masks,  # [T, num_objects, H, W]
            'point_coords': point_coords,  # [num_objects, 1, 2]
            'point_labels': point_labels,  # [num_objects, 1]
            'video_name': video_name,
            'frame_indices': frame_indices,
        }


class PrecomputedEmbeddingDataset(Dataset):
    """
    Dataset that loads pre-computed embeddings instead of raw images.
    
    This dramatically speeds up training by avoiding the image encoder
    forward pass on every iteration.
    """
    
    def __init__(
        self,
        embed_path: str,
        annotation_path: str,
        frames_per_video: int = 8,
        embed_dim: int = 1024,
        embed_size: int = 72,
    ):
        super().__init__()
        self.embed_path = embed_path
        self.frames_per_video = frames_per_video
        self.embed_dim = embed_dim
        self.embed_size = embed_size
        
        # Discover embedding files
        self.videos = self._discover_embeddings()
        
    def _discover_embeddings(self) -> List[str]:
        """Find all video embedding directories."""
        videos = []
        if not os.path.exists(self.embed_path):
            return videos
            
        for name in sorted(os.listdir(self.embed_path)):
            video_dir = os.path.join(self.embed_path, name)
            if os.path.isdir(video_dir):
                videos.append(name)
        return videos
    
    def __len__(self) -> int:
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load pre-computed embeddings for a video clip."""
        video_name = self.videos[idx]
        video_dir = os.path.join(self.embed_path, video_name)
        
        # Load embedding files
        embeddings = []
        for i in range(self.frames_per_video):
            embed_file = os.path.join(video_dir, f'frame_{i:05d}.pt')
            if os.path.exists(embed_file):
                embed = torch.load(embed_file, map_location='cpu')
            else:
                # Dummy if file missing
                embed = torch.zeros(self.embed_dim, self.embed_size, self.embed_size)
            embeddings.append(embed)
        
        embeddings = torch.stack(embeddings, dim=0)  # [T, C, H, W]
        
        return {
            'embeddings': embeddings,
            'video_name': video_name,
        }

class SAVTrunkEmbeddingDataset(Dataset):
    """
    Minimal SA-V dataset that loads **precomputed trunk embeddings** produced by
    `stage2/save_video_embeddings_stage2.py`.
    """

    def __init__(
        self,
        sav_root: str,
        embedding_dir: str,
        num_frames: int = 8,
        frame_skip: int = 4,
        videos: Optional[List[str]] = None,
    ):
        self.sav_root = Path(sav_root)
        self.embedding_dir = Path(embedding_dir)
        self.num_frames = int(num_frames)
        self.frame_skip = int(frame_skip)

        if videos is None:
            self.videos = sorted(
                [
                    d.name
                    for d in self.sav_root.iterdir()
                    if d.is_dir() and d.name.startswith("sav_")
                ]
            )
        else:
            self.videos = list(videos)

        # Cache frame stems per video (use the image files as the authoritative list)
        self._frames_by_video: Dict[str, List[str]] = {}
        for v in self.videos:
            vdir = self.sav_root / v
            frame_files = sorted(
                [
                    p
                    for p in vdir.iterdir()
                    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ]
            )
            self._frames_by_video[v] = [p.stem for p in frame_files]

    def __len__(self) -> int:
        return len(self.videos)

    def _sample_indices(self, n_total: int) -> List[int]:
        max_start = n_total - (self.num_frames * self.frame_skip)
        if max_start <= 0:
            # Video too short: sample uniformly
            idxs = torch.linspace(0, n_total - 1, steps=self.num_frames)
            return [int(i) for i in idxs]
        start = random.randint(0, max_start)
        return [start + i * self.frame_skip for i in range(self.num_frames)]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_name = self.videos[idx]
        frames = self._frames_by_video[video_name]
        if len(frames) == 0:
            raise RuntimeError(f"No frames found for {video_name} under {self.sav_root}")

        frame_indices = self._sample_indices(len(frames))
        frame_stems = [frames[i] for i in frame_indices]

        embeds = []
        for stem in frame_stems:
            p = self.embedding_dir / video_name / f"{stem}.pt"
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing embedding {p}. Did you run stage2/save_video_embeddings_stage2.py?"
                )
            embeds.append(torch.load(p, map_location="cpu"))

        embeddings = torch.stack(embeds, dim=0)  # [T, 1024, 72, 72]
        return {
            "embeddings": embeddings,
            "video_name": video_name,
            "frame_stems": frame_stems,
        }


def build_loader(config, is_train: bool, use_precomputed: bool) -> DataLoader:
    """
    Build a single DataLoader (train or val).

    This matches the callsites in `stage2/train_memory_stage2.py`.
    """
    sav_root = getattr(config.DATA, "SAV_ROOT", "") or getattr(config.DATA, "DATA_PATH", "")
    if not sav_root:
        raise ValueError("DATA.SAV_ROOT must be set (or legacy DATA.DATA_PATH).")

    num_frames = getattr(config.DATA, "NUM_FRAMES", None) or getattr(config.DATA, "FRAMES_PER_VIDEO", 8)
    frame_skip = getattr(config.DATA, "FRAME_SKIP", 4)

    # Deterministic split by sorted video names
    all_videos = sorted([d.name for d in Path(sav_root).iterdir() if d.is_dir() and d.name.startswith("sav_")])
    if len(all_videos) == 0:
        raise RuntimeError(f"No SA-V videos found under {sav_root!r}")

    train_ratio = float(getattr(config.DATA, "TRAIN_SPLIT", 0.95))
    split_idx = max(1, int(len(all_videos) * train_ratio))
    train_videos = all_videos[:split_idx]
    val_videos = all_videos[split_idx:]

    if is_train:
        videos = train_videos
    else:
        videos = val_videos if len(val_videos) > 0 else train_videos[:1]

    if use_precomputed:
        embedding_dir = getattr(config.DATA, "EMBEDDING_DIR", "") or getattr(config.DATA, "EMBED_PATH", "")
        if not embedding_dir:
            raise ValueError("DATA.EMBEDDING_DIR must be set when DATA.USE_PRECOMPUTED=True")
        dataset: Dataset = SAVTrunkEmbeddingDataset(
            sav_root=sav_root,
            embedding_dir=embedding_dir,
            num_frames=num_frames,
            frame_skip=frame_skip,
            videos=videos,
        )
    else:
        # Fall back to the (older) raw-frame loader for now
        dataset = SAVVideoDataset(
            data_path=sav_root,
            frames_per_video=num_frames,
            frame_skip=frame_skip,
            img_size=config.DATA.IMG_SIZE,
            max_objects=getattr(config.DATA, "MAX_OBJECTS", 3),
            use_precomputed=False,
            embed_path="",
            split="train" if is_train else "val",
            mean=getattr(config.DATA, "MEAN", [123.675, 116.28, 103.53]),
            std=getattr(config.DATA, "STD", [58.395, 57.12, 57.375]),
        )

    loader = DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=is_train,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=is_train,
    )
    return loader
