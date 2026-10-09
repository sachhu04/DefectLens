"""
Vision Transformer (ViT) Feature Extractor with Multi-Criteria Token Fusion (MCTF).

Overview:
---------
Wraps a standard Vision Transformer backbone (e.g., DeiT-Small: `deit_small_patch16_224`)
from `timm` and conditionally injects MCTF modules prior to specified transformer encoder
blocks.

Key Responsibilities:
1. Converts an input image (3 x 224 x 224) into non-overlapping patch embeddings (196 patches).
2. Interleaves MCTF reduction layers between transformer blocks according to configuration.
3. Propagates token cluster sizes and the 1-to-1 provenance mapping throughout the forward pass.
4. Exposes `extract_patch_features` to provide reduced feature vectors and index mappings
   tailored for the PatchCore anomaly detection pipeline.
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Any, List, Optional, Tuple
from .mctf import MCTFModule


class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer Feature Extractor with optional MCTF injection.
    
    Args:
        model_name (str): Backbone model identifier in `timm` (e.g., 'vit_deit_small_patch16_224').
        pretrained (bool): Whether to initialize with pretrained ImageNet weights.
        mctf_config (Optional[Dict[str, Any]]): Configuration dictionary for MCTF.
            If None or `enabled=False`, functions as a standard baseline ViT.
    """
    def __init__(self, model_name: str = 'vit_deit_small_patch16_224', pretrained: bool = True,
                 mctf_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model_name = model_name
        self.mctf_config = mctf_config
        
        # Instantiate base Vision Transformer backbone via timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Cache architectural metadata
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.num_patches = self.backbone.patch_embed.num_patches  # 196 for 224x224 / 16x16
        
        self.mctf_modules = nn.ModuleDict()
        
        # Initialize MCTF modules at designated transformer blocks
        if self.mctf_config and self.mctf_config.get('enabled', False):
            num_blocks = len(self.backbone.blocks)
            # Default to applying at all blocks except the last one if layers not explicitly provided
            self.mctf_layers = self.mctf_config.get('layers', list(range(num_blocks - 1)))
            
            for l in self.mctf_layers:
                self.mctf_modules[str(l)] = MCTFModule(
                    reduction_ratio=self.mctf_config.get('reduction_ratio', 0.2),
                    temperature_sim=self.mctf_config.get('temperature_sim', 1.0),
                    temperature_info=self.mctf_config.get('temperature_info', 1.0),
                    temperature_size=self.mctf_config.get('temperature_size', 1.0),
                    use_sim=self.mctf_config.get('use_sim', True),
                    use_info=self.mctf_config.get('use_info', True),
                    use_size=self.mctf_config.get('use_size', True)
                )
        else:
            self.mctf_layers = []
        
    def forward(self, x: torch.Tensor, return_all_blocks: bool = False):
        """
        Custom forward pass executing ViT tokenization and block processing with MCTF injection.
        
        Flow:
        1. Patch Projection: Transforms [B, 3, 224, 224] -> [B, 196, 384]
        2. [CLS] Token Prepending: Yields sequence of 197 tokens
        3. Position Embedding Addition & Dropout
        4. Block Loop: Conditionally triggers MCTF fusion before block execution
        5. Layer Normalization
        
        Args:
            x (torch.Tensor): Input image tensor [B, 3, H, W].
            return_all_blocks (bool): If True, returns features from all intermediate layers.
            
        Returns:
            Tuple of (x, features, token_sizes, mapping) or (x, token_sizes, mapping).
        """
        B = x.shape[0]
        
        # 1. Project image patches into token embeddings
        x = self.backbone.patch_embed(x)
        
        # 2. Prepend learnable [CLS] classification token
        if self.backbone.cls_token is not None:
            cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
        # 3. Add 1D learnable positional embeddings
        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed
            
        x = self.backbone.pos_drop(x)
        
        num_tokens = x.shape[1]
        
        # Track initial cluster size for each token (1 patch per token)
        token_sizes = torch.ones((B, num_tokens, 1), device=x.device, dtype=x.dtype)
        # Identity provenance mapping: mapping[b, i] = i
        mapping = torch.arange(num_tokens, device=x.device).unsqueeze(0).expand(B, -1).clone()
        
        features = []
        num_blocks = len(self.backbone.blocks)
        
        # 4. Sequentially process transformer encoder blocks
        for i, block in enumerate(self.backbone.blocks):
            # Apply MCTF token fusion if this layer is configured
            if str(i) in self.mctf_modules:
                next_block = self.backbone.blocks[i+1] if i+1 < num_blocks else None
                mctf = self.mctf_modules[str(i)]
                # Fuses tokens, updates token sizes and spatial provenance mapping
                x, token_sizes, mapping = mctf(x, token_sizes, mapping, next_block)
            
            # Execute standard Transformer Block (MHA + MLP + LayerNorm + Residuals)
            x = block(x)
            features.append(x)
            
        # Final LayerNorm
        x = self.backbone.norm(x)
        
        if return_all_blocks:
            return x, features, token_sizes, mapping
        return x, token_sizes, mapping

    def extract_patch_features(self, x: torch.Tensor, layer_idx: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method to retrieve patch representations and the spatial mapping
        from a specific layer (e.g., layer_idx=-2, the penultimate block).
        
        Note:
        - The [CLS] token is stripped from the returned features.
        - Provenance mapping indices are shifted by -1 so that they cleanly index into
          the patch-only feature tensor (range [0, N_reduced-1]).
          
        Args:
            x (torch.Tensor): Input batch [B, 3, 224, 224].
            layer_idx (int): Block index from which features are extracted (default -1).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - feat: Patch feature tensor [B, N_reduced, C].
                - mapping: Original patch to reduced token index mapping [B, N_orig].
        """
        final_x, features, sizes, mapping = self.forward(x, return_all_blocks=True)
        feat = features[layer_idx]
        
        # Exclude [CLS] token at index 0 from patch feature bank and mapping
        if self.backbone.cls_token is not None:
            feat = feat[:, 1:]
            # mapping[b, 1:] provides token index for original patches 1..N
            # Subtract 1 because index 0 ([CLS]) has been removed from feat
            mapping = mapping[:, 1:] - 1
            
        return feat, mapping
