# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Adapted from EdgeTAM's Perceiver implementation for EfficientSAM3

"""
Perceiver Resampler for efficient memory compression.

This module implements the Perceiver Resampler from EdgeTAM paper which
compresses dense memory features (HxW tokens per frame) into a small set
of learnable latent vectors, significantly reducing memory attention cost.

The module supports two types of latents:
- Global latents: Attend to the entire feature map globally
- 2D Spatial latents: Attend to local windows with explicit position encoding

This combines the strengths of both approaches:
- Global: Captures semantic information across the whole frame
- 2D Spatial: Preserves local spatial structure needed for dense prediction
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def FeedForward(dim: int, mult: int = 4) -> nn.Sequential:
    """Simple FFN block with LayerNorm, Linear, GELU, Linear."""
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    """
    Cross-attention from learnable latents to input features.
    
    Latents serve as queries, input features as keys/values.
    Optionally concatenates latents to keys/values for self-attention effect.
    """
    
    def __init__(
        self,
        *,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout_p: float = 0.05,
        concat_kv_latents: bool = False,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_x = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        # Position encoding projection (same as key projection)
        self.to_pos = nn.Linear(dim, inner_dim, bias=False) if dim != inner_dim else nn.Identity()

        self.dropout_p = dropout_p
        self.concat_kv_latents = concat_kv_latents

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(
        self,
        latents: torch.Tensor,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latents: [B, N_latents, C] - learnable query tokens
            x: [B, N_input, C] - input features (memory)
            pos: [B, N_input, C] - optional positional encoding for keys/values
        
        Returns:
            [B, N_latents, C] - updated latent tokens
        """
        latents = self.norm_latents(latents)
        x = self.norm_x(x)

        q = self.to_q(latents)

        # Optionally concat latents to key/values for self-attention effect
        if self.concat_kv_latents:
            kv_input = torch.cat((x, latents), dim=-2)
        else:
            kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = self._separate_heads(q, self.heads)
        k = self._separate_heads(k, self.heads)
        v = self._separate_heads(v, self.heads)

        # Add positional encoding to keys and values
        if pos is not None:
            assert not self.concat_kv_latents, "Cannot use pos_enc with concat_kv_latents"
            pos = self.to_pos(pos)  # Project to inner_dim
            pos = self._separate_heads(pos, self.heads)
            k, v = k + pos, v + pos

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = self._recombine_heads(out)
        return self.to_out(out)


class SelfAttention(nn.Module):
    """Standard self-attention for latent tokens."""
    
    def __init__(
        self,
        *,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout_p: float = 0.05,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout_p = dropout_p

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q = self._separate_heads(q, self.heads)
        k = self._separate_heads(k, self.heads)
        v = self._separate_heads(v, self.heads)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = self._recombine_heads(out)
        return self.to_out(out)


class PerceiverEncoderLayer(nn.Module):
    """
    Single Perceiver encoder layer with cross-attention and optional self-attention.
    
    Architecture:
        latents -> CrossAttn(latents, input) -> FFN -> [SelfAttn -> FFN] -> latents
    """
    
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        hidden_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.0,
        concat_kv_latents: bool = False,
        use_self_attn: bool = True,
    ):
        super().__init__()
        self.attn = PerceiverAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout_p=attention_dropout_p,
            concat_kv_latents=concat_kv_latents,
        )
        self.ff = FeedForward(dim=dim, mult=ff_mult)
        self.dropout = nn.Dropout(hidden_dropout_p)
        self.use_self_attn = use_self_attn
        
        if use_self_attn:
            self.self_attn = SelfAttention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                dropout_p=attention_dropout_p,
            )
            self.self_ff = FeedForward(dim=dim, mult=ff_mult)

    def forward(
        self,
        latents: torch.Tensor,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Cross-attention: latents attend to input
        latents = self.attn(latents, x, pos) + latents
        latents = self.dropout(latents)
        latents = self.ff(latents) + latents
        
        # Optional self-attention among latents
        if self.use_self_attn:
            latents = self.self_attn(latents) + latents
            latents = self.self_ff(latents) + latents
        
        return latents


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.
    
    Args:
        x: (B, H, W, C) feature map
        window_size: Size of each window
        
    Returns:
        windows: (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition.
    
    Args:
        windows: (num_windows * B, window_size, window_size, C)
        window_size: Size of each window
        H, W: Original height and width
        
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler for memory compression (from EdgeTAM).
    
    Compresses dense memory features into a small set of latent tokens:
    - Global latents (num_latents): Attend globally to entire feature map
    - 2D Spatial latents (num_latents_2d): Attend to local windows
    
    This significantly reduces the number of memory tokens from HxW to
    num_latents + num_latents_2d, enabling efficient memory attention.
    
    Args:
        dim: Feature dimension
        depth: Number of encoder layers
        dim_head: Dimension per attention head
        heads: Number of attention heads
        num_latents: Number of global latent tokens
        num_latents_2d: Number of 2D spatial latent tokens
        ff_mult: FFN expansion factor
        hidden_dropout_p: Dropout for hidden layers
        attention_dropout_p: Dropout for attention
        pos_enc_at_key_value: Add pos encoding to keys/values
        use_self_attn: Use self-attention in encoder layers
        position_encoding: Module to generate position encoding for 2D latents
    """
    
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 2,
        dim_head: int = 64,
        heads: int = 1,
        num_latents: int = 256,
        num_latents_2d: int = 256,
        ff_mult: int = 4,
        hidden_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.05,
        pos_enc_at_key_value: bool = False,
        concat_kv_latents: bool = False,
        position_encoding: Optional[nn.Module] = None,
        use_self_attn: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
        self.num_latents_2d = num_latents_2d
        
        # Initialize learnable latent tokens
        if num_latents > 0:
            self.latents = nn.Parameter(torch.randn(num_latents, dim))
        else:
            self.register_parameter('latents', None)
            
        if num_latents_2d > 0:
            self.latents_2d = nn.Parameter(torch.randn(num_latents_2d, dim))
        else:
            self.register_parameter('latents_2d', None)
            
        self.position_encoding = position_encoding
        self.pos_enc_at_key_value = pos_enc_at_key_value

        # Encoder layers (shared between global and 2D paths)
        self.layers = nn.ModuleList([
            PerceiverEncoderLayer(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                hidden_dropout_p=hidden_dropout_p,
                attention_dropout_p=attention_dropout_p,
                concat_kv_latents=concat_kv_latents,
                use_self_attn=use_self_attn,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compress memory features using Perceiver.
        
        Args:
            x: [B, C, H, W] - input memory features
            pos: [B, C, H, W] - optional positional encoding
            
        Returns:
            latents: [B, N_total, C] - compressed latent tokens
            pos_out: [B, N_total, C] - positional encoding for latents (or None)
        """
        out_latents = []
        out_pos = []
        
        if self.num_latents > 0:
            latents_1d, pos_1d = self._forward_global(x, pos)
            out_latents.append(latents_1d)
            out_pos.append(pos_1d)
            
        if self.num_latents_2d > 0:
            latents_2d, pos_2d = self._forward_2d_spatial(x)
            out_latents.append(latents_2d)
            out_pos.append(pos_2d)

        latents = torch.cat(out_latents, dim=1)
        
        if pos is not None or self.num_latents_2d > 0:
            pos_out = torch.cat(out_pos, dim=1)
        else:
            pos_out = None

        return latents, pos_out

    def _forward_global(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process with global latents that attend to entire feature map.
        
        Args:
            x: [B, C, H, W]
            pos: [B, C, H, W] or None
            
        Returns:
            latents: [B, num_latents, C]
            pos: [B, num_latents, C] or None
        """
        B = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, N, C]
        
        # Flatten spatial dims: [B, C, H, W] -> [B, HW, C]
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        
        # Prepare positional encoding
        _pos = None
        if self.pos_enc_at_key_value and pos is not None:
            _pos = pos.permute(0, 2, 3, 1).flatten(1, 2)

        # Apply encoder layers
        for layer in self.layers:
            latents = layer(latents, x, _pos)

        latents = self.norm(latents)
        
        # Output positional encoding is zeros for global latents
        if pos is not None:
            pos_out = torch.zeros_like(latents)
        else:
            pos_out = None
            
        return latents, pos_out

    def _forward_2d_spatial(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process with 2D spatial latents using window attention.
        
        Each latent attends to a local window, preserving spatial structure.
        
        Args:
            x: [B, C, H, W]
            
        Returns:
            latents: [B, num_latents_2d, C]
            pos: [B, num_latents_2d, C]
        """
        B, C, H, W = x.shape
        
        # Calculate window grid size based on number of 2D latents
        num_window = int(math.sqrt(self.num_latents_2d))
        
        # Resize input to be divisible by num_window using adaptive pooling
        # This handles cases where H is not divisible by num_window
        target_size = num_window * (H // num_window) if H % num_window == 0 else num_window * 4
        if H != target_size:
            x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
        
        window_size = target_size // num_window
        
        # Prepare latents: one per window
        latents_2d = self.latents_2d.unsqueeze(0).expand(B, -1, -1)  # [B, N, C]
        latents_2d = latents_2d.reshape(B * self.num_latents_2d, 1, C)  # [B*N, 1, C]
        
        # Partition into windows
        x_spatial = x.permute(0, 2, 3, 1)  # [B, H', W', C]
        x_windows = window_partition(x_spatial, window_size)  # [B*num_windows, ws, ws, C]
        x_windows = x_windows.flatten(1, 2)  # [B*num_windows, ws*ws, C]

        # Apply encoder layers (each latent attends to its window)
        for layer in self.layers:
            latents_2d = layer(latents_2d, x_windows)

        # Reshape back to [B, N, C]
        latents_2d = latents_2d.reshape(B, self.num_latents_2d, C)

        # Generate position encoding for 2D latents as spatial grid
        # Create a grid of positions
        grid_size = num_window
        if self.position_encoding is not None:
            # Create dummy spatial tensor for position encoding
            dummy_spatial = latents_2d.reshape(B, num_window, num_window, C).permute(0, 3, 1, 2)
            pos_2d = self.position_encoding(dummy_spatial)
            pos_2d = pos_2d.permute(0, 2, 3, 1).flatten(1, 2)  # [B, N, C]
        else:
            pos_2d = torch.zeros_like(latents_2d)

        latents_2d = self.norm(latents_2d)

        return latents_2d, pos_2d

class EfficientSpatialPerceiver(nn.Module):
    """
    Wrapper that combines Perceiver compression with the memory encoder.
    
    This module is inserted after the memory encoder to compress the
    spatial memory features before they enter the memory bank.
    
    Memory flow:
        visual_features + mask -> MemoryEncoder -> SpatialPerceiver -> compressed_memory
        
    The compressed memory has shape [B, N_latents, C] instead of [B, C, H, W],
    reducing memory attention complexity from O(HW^2) to O(N^2).
    """
    
    def __init__(
        self,
        dim: int = 64,
        num_latents: int = 256,
        num_latents_2d: int = 256,
        depth: int = 2,
        heads: int = 1,
        dim_head: int = 64,
        use_self_attn: bool = True,
        position_encoding: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.perceiver = PerceiverResampler(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_latents=num_latents,
            num_latents_2d=num_latents_2d,
            ff_mult=4,
            hidden_dropout_p=0.0,  # No dropout during inference
            attention_dropout_p=0.0,
            pos_enc_at_key_value=True,
            use_self_attn=use_self_attn,
            position_encoding=position_encoding,
        )
        self.num_latents = num_latents
        self.num_latents_2d = num_latents_2d

    def forward(
        self,
        maskmem_features: torch.Tensor,
        maskmem_pos_enc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress memory features using Perceiver.
        
        Args:
            maskmem_features: [B, C, H, W] - output from memory encoder
            maskmem_pos_enc: [B, C, H, W] - positional encoding
            
        Returns:
            compressed_features: [B, N, C] - compressed memory tokens
            compressed_pos_enc: [B, N, C] - positional encoding for tokens
        """
        compressed_features, compressed_pos_enc = self.perceiver(
            maskmem_features, maskmem_pos_enc
        )
        return compressed_features, compressed_pos_enc
