import torch
import torch.nn as nn
import timm
from typing import Dict, Any, List, Optional, Tuple
from .mctf import MCTFModule

class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = 'vit_deit_small_patch16_224', pretrained: bool = True,
                 mctf_config: Optional[Dict[str, Any]] = None):
        """
        ViT Feature Extractor that can optionally integrate MCTF.
        Args:
            model_name: timm model name.
            pretrained: whether to load pretrained weights.
            mctf_config: Configuration dictionary for MCTF. If None, MCTF is disabled.
        """
        super().__init__()
        self.model_name = model_name
        self.mctf_config = mctf_config
        
        # Load the base model
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Determine patch size and embed dim
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.num_patches = self.backbone.patch_embed.num_patches
        
        self.mctf_modules = nn.ModuleDict()
        
        if self.mctf_config and self.mctf_config.get('enabled', False):
            # Configuration for MCTF layers (e.g. at which block indices to apply)
            # Default to applying at all blocks except the last one.
            num_blocks = len(self.backbone.blocks)
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
        Custom forward pass that supports MCTF injection.
        """
        B = x.shape[0]
        x = self.backbone.patch_embed(x)
        
        if self.backbone.cls_token is not None:
            cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed
            
        x = self.backbone.pos_drop(x)
        
        num_tokens = x.shape[1]
        token_sizes = torch.ones((B, num_tokens, 1), device=x.device, dtype=x.dtype)
        mapping = torch.arange(num_tokens, device=x.device).unsqueeze(0).expand(B, -1).clone()
        
        features = []
        num_blocks = len(self.backbone.blocks)
        
        for i, block in enumerate(self.backbone.blocks):
            if str(i) in self.mctf_modules:
                next_block = self.backbone.blocks[i+1] if i+1 < num_blocks else None
                mctf = self.mctf_modules[str(i)]
                x, token_sizes, mapping = mctf(x, token_sizes, mapping, next_block)
            
            x = block(x)
            features.append(x)
            
        x = self.backbone.norm(x)
        
        if return_all_blocks:
            return x, features, token_sizes, mapping
        return x, token_sizes, mapping

    def extract_patch_features(self, x: torch.Tensor, layer_idx: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method to get patch features and original patch mapping from a specific layer (default last).
        """
        final_x, features, sizes, mapping = self.forward(x, return_all_blocks=True)
        feat = features[layer_idx]
        
        # Remove cls token if present (both from feat and mapping)
        if self.backbone.cls_token is not None:
            feat = feat[:, 1:]
            # mapping[b, 1:] gives the index of the token representing original patch 1..N
            # We subtract 1 from these indices because we removed the CLS token at index 0 in feat.
            mapping = mapping[:, 1:] - 1
            
        return feat, mapping
