# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Adapted from SAM2/EdgeTAM/EfficientTAM for EfficientSAM3

"""
Efficient Memory Attention Module for EfficientSAM3.

This module implements an efficient version of SAM3's memory attention by:
1. Using 2x2 average pooling on spatial memory tokens (from EfficientTAM)
2. Reducing from 4 to 2 transformer layers (from EdgeTAM)
3. Keeping object pointer tokens unpooled for precise object tracking

The key insight from EfficientTAM is that spatial memory tokens exhibit strong
locality (neighboring tokens are similar), allowing efficient compression
while maintaining cross-attention quality.

Complexity reduction:
- Original: O(T * C * H^2 * W^2) where T=7 frames, H=W=72
- Efficient: O(T * C * H * W * (H/2 * W/2)) = 4x reduction in attention
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(activation: str):
    """Return activation function from string name."""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation: {activation}")


class RoPE2D(nn.Module):
    """
    2D Rotary Position Encoding for attention.
    
    Applies rotary position encoding along both spatial dimensions,
    which helps the model understand spatial relationships.
    """
    
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        
    def _compute_freqs(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute frequency tensor for RoPE."""
        freqs = 1.0 / (self.theta ** (
            torch.arange(0, self.dim, 2, device=device).float() / self.dim
        ))
        t = torch.arange(seq_len, device=device)
        freqs = torch.einsum('i,j->ij', t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        feat_size: tuple,
    ) -> tuple:
        """Apply RoPE to queries and keys."""
        # For simplicity, we skip RoPE implementation here
        # In production, use the full SAM2 RoPE implementation
        return q, k


class EfficientCrossAttention(nn.Module):
    """
    Efficient cross-attention with spatial token pooling.
    
    From EfficientTAM: neighboring spatial tokens are similar due to local
    smoothness in visual features. We exploit this by:
    1. Average pooling spatial keys/values with 2x2 windows
    2. Adding log(pool_size) to attention logits to compensate
    3. Keeping object pointer tokens unpooled
    
    This reduces attention complexity by ~4x while maintaining quality.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        pool_size: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool_size = pool_size
        self.pool_compensation = math.log(pool_size * pool_size)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, N, C) -> (B, num_heads, N, head_dim)"""
        B, N, C = x.shape
        x = x.reshape(B, N, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, num_heads, N, head_dim) -> (B, N, C)"""
        B, _, N, _ = x.shape
        x = x.transpose(1, 2)
        return x.reshape(B, N, self.embed_dim)

    def _pool_spatial_tokens(
        self,
        tokens: torch.Tensor,
        num_spatial: int,
        H: int,
        W: int,
    ) -> tuple:
        """
        Pool only the spatial tokens, keep object pointer tokens.
        
        Args:
            tokens: [B, N_spatial + N_ptr, C] - all memory tokens
            num_spatial: Number of spatial tokens (H * W per frame)
            H, W: Spatial dimensions
            
        Returns:
            pooled: [B, N_pooled + N_ptr, C]
            pool_info: dict with pooling metadata
        """
        B, N_total, C = tokens.shape
        N_ptr = N_total - num_spatial
        
        # Split spatial and pointer tokens
        spatial_tokens = tokens[:, :num_spatial, :]  # [B, num_spatial, C]
        ptr_tokens = tokens[:, num_spatial:, :]  # [B, N_ptr, C]
        
        # Determine number of frames
        tokens_per_frame = H * W
        num_frames = num_spatial // tokens_per_frame
        
        # Reshape spatial tokens: [B, num_frames, H, W, C]
        spatial_tokens = spatial_tokens.view(B, num_frames, H, W, C)
        
        # Apply 2x2 average pooling per frame
        # Reshape for pooling: [B * num_frames, C, H, W]
        spatial_tokens = spatial_tokens.permute(0, 1, 4, 2, 3)  # [B, F, C, H, W]
        spatial_tokens = spatial_tokens.reshape(B * num_frames, C, H, W)
        
        pooled_spatial = F.avg_pool2d(
            spatial_tokens,
            kernel_size=self.pool_size,
            stride=self.pool_size,
        )
        
        # Reshape back: [B, F, C, H', W'] -> [B, F * H' * W', C]
        _, _, H_pool, W_pool = pooled_spatial.shape
        pooled_spatial = pooled_spatial.view(B, num_frames, C, H_pool, W_pool)
        pooled_spatial = pooled_spatial.permute(0, 1, 3, 4, 2)  # [B, F, H', W', C]
        pooled_spatial = pooled_spatial.reshape(B, num_frames * H_pool * W_pool, C)
        
        # Concatenate pooled spatial with original pointer tokens
        pooled = torch.cat([pooled_spatial, ptr_tokens], dim=1)
        
        pool_info = {
            'H': H,
            'W': W,
            'H_pool': H_pool,
            'W_pool': W_pool,
            'num_frames': num_frames,
            'num_spatial_pooled': num_frames * H_pool * W_pool,
            'num_ptr': N_ptr,
        }
        
        return pooled, pool_info

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_spatial_mem: int = -1,
        feat_size: tuple = (72, 72),
        use_pooling: bool = True,
    ) -> torch.Tensor:
        """
        Efficient cross-attention with optional spatial pooling.
        
        Args:
            q: [B, N_q, C] - query tokens (current frame features)
            k: [B, N_kv, C] - key tokens (memory features)
            v: [B, N_kv, C] - value tokens (memory features)
            num_spatial_mem: Number of spatial memory tokens
            feat_size: (H, W) spatial dimensions
            use_pooling: Whether to apply pooling
            
        Returns:
            out: [B, N_q, C]
        """
        B = q.shape[0]
        H, W = feat_size
        tokens_per_frame = H * W
        
        # Project queries
        q = self.q_proj(q)
        
        # Check if memory is already compressed (less than full resolution per frame)
        # If num_spatial_mem is not divisible by H*W, memory is already compressed
        can_pool = (num_spatial_mem > 0 and 
                    num_spatial_mem >= tokens_per_frame and
                    num_spatial_mem % tokens_per_frame == 0)
        
        # Pool spatial keys/values if enabled and possible
        if use_pooling and can_pool:
            k_pooled, pool_info = self._pool_spatial_tokens(k, num_spatial_mem, H, W)
            v_pooled, _ = self._pool_spatial_tokens(v, num_spatial_mem, H, W)
            
            k = self.k_proj(k_pooled)
            v = self.v_proj(v_pooled)
            
            # Separate into heads
            q = self._separate_heads(q)
            k = self._separate_heads(k)
            v = self._separate_heads(v)
            
            # Compute attention with pooling compensation
            # Add log(pool_size^2) to pooled spatial attention logits
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Compensate for pooling (only for spatial tokens)
            n_spatial_pooled = pool_info['num_spatial_pooled']
            attn_logits[:, :, :, :n_spatial_pooled] += self.pool_compensation
            
            attn = F.softmax(attn_logits, dim=-1)
            attn = self.dropout(attn)
            
            out = torch.matmul(attn, v)
        else:
            # Standard attention without pooling
            k = self.k_proj(k)
            v = self.v_proj(v)
            
            q = self._separate_heads(q)
            k = self._separate_heads(k)
            v = self._separate_heads(v)
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        
        return out


class EfficientMemoryAttentionLayer(nn.Module):
    """
    Single layer of efficient memory attention.
    
    Architecture:
        Self-Attention -> Cross-Attention -> FFN
        
    Cross-attention uses efficient pooling for memory tokens.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        dim_feedforward: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.0,
        activation: str = "gelu",
        pool_size: int = 2,
        pos_enc_at_attn: bool = True,
        pos_enc_at_cross_attn_keys: bool = True,
        pos_enc_at_cross_attn_queries: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Efficient cross-attention with pooling
        self.cross_attn = EfficientCrossAttention(
            d_model, num_heads, dropout=dropout, pool_size=pool_size
        )
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = get_activation_fn(activation)
        
        # Position encoding options
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries

    def _forward_sa(
        self,
        tgt: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Self-attention on current frame features."""
        tgt2 = self.norm1(tgt)
        if self.pos_enc_at_attn and query_pos is not None:
            q = k = tgt2 + query_pos
        else:
            q = k = tgt2
        tgt2, _ = self.self_attn(q, k, tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        num_spatial_mem: int = -1,
        feat_size: tuple = (72, 72),
    ) -> torch.Tensor:
        """Cross-attention from current frame to memory."""
        tgt2 = self.norm2(tgt)
        
        # Prepare queries
        if self.pos_enc_at_cross_attn_queries and query_pos is not None:
            q = tgt2 + query_pos
        else:
            q = tgt2
            
        # Prepare keys/values with optional position encoding
        if self.pos_enc_at_cross_attn_keys and memory_pos is not None:
            k = memory + memory_pos
        else:
            k = memory
        v = memory
        
        tgt2 = self.cross_attn(
            q, k, v,
            num_spatial_mem=num_spatial_mem,
            feat_size=feat_size,
            use_pooling=num_spatial_mem > 0,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def _forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        """Feedforward network."""
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        num_spatial_mem: int = -1,
        feat_size: tuple = (72, 72),
    ) -> torch.Tensor:
        """
        Full forward pass through the layer.
        
        Args:
            tgt: [B, N, C] - current frame features (queries)
            memory: [B, M, C] - memory features (keys/values)
            pos: [B, M, C] - memory positional encoding
            query_pos: [B, N, C] - query positional encoding
            num_spatial_mem: Number of spatial memory tokens (for pooling)
            feat_size: (H, W) spatial dimensions
            
        Returns:
            tgt: [B, N, C] - updated features
        """
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(
            tgt, memory, query_pos, pos,
            num_spatial_mem=num_spatial_mem,
            feat_size=feat_size,
        )
        tgt = self._forward_ffn(tgt)
        return tgt


class EfficientMemoryAttention(nn.Module):
    """
    Efficient Memory Attention module for video object tracking.
    
    This is a drop-in replacement for SAM3's TransformerEncoderCrossAttention
    with the following improvements:
    1. 2x2 spatial pooling of memory keys/values (from EfficientTAM)
    2. Reduced number of layers (4 -> 2) (from EdgeTAM)
    3. Support for both dense features and compressed Perceiver latents
    
    When used with Perceiver-compressed memories, pooling is disabled
    since the tokens are already compressed.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        dim_feedforward: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,  # Reduced from 4
        dropout: float = 0.0,
        activation: str = "gelu",
        pool_size: int = 2,
        pos_enc_at_input: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_enc_at_input = pos_enc_at_input
        
        # For compatibility with Sam3TrackerBase which checks `transformer.decoder is None`
        self.decoder = None
        
        # Build layers
        self.layers = nn.ModuleList([
            EfficientMemoryAttentionLayer(
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
                pool_size=pool_size,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        num_obj_ptr_tokens: int = 0,
        num_spatial_mem: int = -1,
        feat_size: tuple = (72, 72),
    ) -> torch.Tensor:
        """
        Fuse current frame features with memory.
        
        Args:
            curr: [HW, B, C] or [B, HW, C] - current frame features
            memory: [M, B, C] or [B, M, C] - memory features
            curr_pos: Positional encoding for current frame
            memory_pos: Positional encoding for memory
            num_obj_ptr_tokens: Number of object pointer tokens
            num_spatial_mem: Total number of spatial memory tokens
            feat_size: (H, W) spatial dimensions
            
        Returns:
            output: [HW, B, C] or [B, HW, C] - memory-conditioned features
        """
        # Handle list inputs (from SAM3 backbone)
        if isinstance(curr, list):
            assert len(curr) == 1
            curr = curr[0]
        if isinstance(curr_pos, list):
            assert len(curr_pos) == 1
            curr_pos = curr_pos[0]
            
        # Determine if input is (seq, batch, dim) or (batch, seq, dim)
        batch_first = curr.shape[0] != feat_size[0] * feat_size[1]
        
        # Convert to batch-first if needed
        if not batch_first:
            curr = curr.transpose(0, 1)  # [B, HW, C]
            if curr_pos is not None:
                curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            if memory_pos is not None:
                memory_pos = memory_pos.transpose(0, 1)

        # Optional: add position encoding at input
        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        # Apply transformer layers
        for layer in self.layers:
            output = layer(
                output,
                memory,
                pos=memory_pos,
                query_pos=curr_pos,
                num_spatial_mem=num_spatial_mem,
                feat_size=feat_size,
            )

        output = self.norm(output)

        # Convert back to original format
        if not batch_first:
            output = output.transpose(0, 1)  # [HW, B, C]

        return output
