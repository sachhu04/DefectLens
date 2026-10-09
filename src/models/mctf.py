"""
Multi-Criteria Token Fusion (MCTF) Module for Vision Transformers.

Overview:
---------
Vision Transformers (ViT) divide an input image into a sequence of non-overlapping
patches. For an input of size 224x224 with 16x16 patch size, this yields 196 patch tokens
(plus 1 [CLS] token = 197 tokens total). Computing self-attention scales quadratically
O(N^2) with the number of tokens across all transformer layers.

MCTF drastically accelerates inference by progressively fusing redundant tokens inside
intermediate transformer layers without requiring retraining. Unlike naive pruning
methods that discard tokens entirely (destroying spatial context for localization),
MCTF merges similar tokens using a weighted average and maintains a 1-to-1 provenance
mapping back to the original 14x14 patch grid.

Fusion Policy (Tripartite Criteria):
------------------------------------
Pairs of tokens are evaluated for fusion using three complementary criteria:
1. Cosine Similarity (W_sim):
   Tokens with similar feature representations (e.g., uniform background regions)
   are prioritized for merging.
2. Informativeness (W_info):
   Tokens receiving high attention weights from the subsequent transformer block
   indicate salient regions or defects and are protected from fusion.
3. Cluster Size Balancing (W_size):
   Inversely weights the combined patch count of candidate tokens to prevent a single
   token from swallowing too many patches (token monopolization).

Composite Fusion Weight:
   W = (W_sim ^ tau_sim) * (W_info ^ tau_info) * (W_size ^ tau_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MCTFModule(nn.Module):
    """
    Applies Multi-Criteria Token Fusion between ViT transformer blocks.
    
    Args:
        reduction_ratio (float): Fraction of current patch tokens to fuse at this stage (e.g., 0.2 = 20%).
        temperature_sim (float): Exponent scaling the similarity criterion (tau_sim).
        temperature_info (float): Exponent scaling the informativeness criterion (tau_info).
        temperature_size (float): Exponent scaling the cluster size criterion (tau_size).
        use_sim (bool): Whether to enable feature similarity in scoring.
        use_info (bool): Whether to enable attention-based informativeness in scoring.
        use_size (bool): Whether to enable cluster size balancing in scoring.
    """
    def __init__(self, 
                 reduction_ratio: float = 0.2,
                 temperature_sim: float = 1.0,
                 temperature_info: float = 1.0,
                 temperature_size: float = 1.0,
                 use_sim: bool = True,
                 use_info: bool = True,
                 use_size: bool = True):
        super().__init__()
        self.r = reduction_ratio
        self.tau_sim = temperature_sim
        self.tau_info = temperature_info
        self.tau_size = temperature_size
        self.use_sim = use_sim
        self.use_info = use_info
        self.use_size = use_size

    def get_informativeness(self, x: torch.Tensor, next_block: nn.Module) -> torch.Tensor:
        """
        Computes token informativeness using the attention weights of the upcoming transformer block.
        
        Tokens that are heavily attended to by other tokens serve as primary hubs of information
        (e.g., object contours, surface defects) and should be preserved.
        
        Args:
            x (torch.Tensor): Current token representations of shape [B, N, C].
            next_block (nn.Module): The subsequent transformer encoder block containing .attn.
            
        Returns:
            torch.Tensor: Normalized informativeness scores of shape [B, N, 1].
        """
        # If disabled or this is the final block, treat all tokens as equally informative
        if not self.use_info or next_block is None:
            return torch.ones((x.shape[0], x.shape[1], 1), device=x.device, dtype=x.dtype)
            
        B, N, C = x.shape
        attn = next_block.attn
        
        # Project tokens into Q, K, V representations
        # Shape: [3, B, num_heads, N, head_dim]
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention probabilities: [B, num_heads, N, N]
        attn_scores = (q @ k.transpose(-2, -1)) * attn.scale
        attn_probs = attn_scores.softmax(dim=-1)
        
        # Average across all attention heads: [B, N, N]
        # Summing across queries (dim=1 of columns) yields total attention received by each key token
        info = attn_probs.mean(dim=1).sum(dim=1, keepdim=True) # [B, 1, N]
        info = info.transpose(1, 2)                            # [B, N, 1]
        
        # Normalize into [0, 1] per sample for stable temperature weighting
        info = info / (info.max(dim=1, keepdim=True)[0] + 1e-8)
        return info

    def forward(self, x: torch.Tensor, token_sizes: torch.Tensor, mapping: torch.Tensor = None, next_block: nn.Module = None) -> tuple:
        """
        Executes bipartite token matching, multi-criteria fusion, and spatial mapping updates.
        
        Args:
            x (torch.Tensor): Token tensor of shape [B, N, C] where index 0 is [CLS].
            token_sizes (torch.Tensor): Number of original patches bundled in each token [B, N, 1].
            mapping (torch.Tensor, optional): Tracks original patch index -> current token index [B, N_orig].
            next_block (nn.Module, optional): Following block used for informativeness calculation.
            
        Returns:
            final_out (torch.Tensor): Reduced token tensor [B, N_reduced, C].
            final_sizes (torch.Tensor): Updated token patch cluster sizes [B, N_reduced, 1].
            final_mapping (torch.Tensor): Updated provenance mapping [B, N_orig].
        """
        B, N, C = x.shape
        # Compute exact number of patch tokens to merge based on reduction ratio 'r'
        num_reduce = int((N - 1) * self.r)
        
        # Initialize identity mapping if not provided: patch i maps to token i
        if mapping is None:
            mapping = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1).clone()

        # If reduction budget is zero, pass through unchanged
        if num_reduce <= 0:
            return x, token_sizes, mapping
            
        # ------------------------------------------------------------------
        # Step 1: Isolate [CLS] token from spatial patch tokens
        # [CLS] token holds global classification context and is NEVER merged.
        # ------------------------------------------------------------------
        cls_token = x[:, 0:1, :]
        cls_size = token_sizes[:, 0:1, :]
        
        tokens = x[:, 1:, :]           # [B, N-1, C]
        sizes = token_sizes[:, 1:, :]   # [B, N-1, 1]
        num_patches = tokens.shape[1]
        
        # ------------------------------------------------------------------
        # Step 2: Bipartite Graph Partitioning (Set A and Set B)
        # Even indices form partition A (sources); Odd indices form partition B (destinations).
        # This ensures efficient O(N_A * N_B) matching without self-pairing.
        # ------------------------------------------------------------------
        a_idx = torch.arange(0, num_patches, 2, device=x.device)
        b_idx = torch.arange(1, num_patches, 2, device=x.device)
        
        tokens_a = tokens[:, a_idx, :]
        tokens_b = tokens[:, b_idx, :]
        sizes_a = sizes[:, a_idx, :]
        sizes_b = sizes[:, b_idx, :]
        
        # ------------------------------------------------------------------
        # Step 3: Compute Multi-Criteria Fusion Scores
        # ------------------------------------------------------------------
        # Criterion 1: Cosine Feature Similarity (mapped to [0, 1])
        if self.use_sim:
            norm_a = F.normalize(tokens_a, dim=-1)
            norm_b = F.normalize(tokens_b, dim=-1)
            sim = (norm_a @ norm_b.transpose(-2, -1) + 1.0) / 2.0  # [B, len(A), len(B)]
            W_sim = sim ** self.tau_sim
        else:
            W_sim = 1.0
            
        # Criterion 2: Informativeness Protection
        # Tokens with high attention info_a + info_b produce low W_info, protecting them from fusion.
        if self.use_info:
            info = self.get_informativeness(x, next_block)
            info_a = info[:, 1:, :][:, a_idx, :]
            info_b = info[:, 1:, :][:, b_idx, :]
            info_sum = info_a + info_b.transpose(1, 2)             # [B, len(A), len(B)]
            W_info = torch.exp(-info_sum) ** self.tau_info
        else:
            W_info = 1.0
            
        # Criterion 3: Cluster Size Balancing
        # Inversely weights candidate pair sizes: larger aggregates have lower fusion priority.
        if self.use_size:
            size_sum = sizes_a + sizes_b.transpose(1, 2)           # [B, len(A), len(B)]
            W_size = (1.0 / size_sum) ** self.tau_size
        else:
            W_size = 1.0
            
        # Composite affinity matrix: [B, len(A), len(B)]
        W = W_sim * W_info * W_size
        
        # ------------------------------------------------------------------
        # Step 4: Bipartite Matching & Top-K Selection
        # For each source token in A, find its best destination candidate in B.
        # ------------------------------------------------------------------
        max_W, max_idx = W.max(dim=-1)                       # Best destination in B for each token in A
        topk_W, topk_a_idx = max_W.topk(num_reduce, dim=-1)  # Top 'num_reduce' source tokens to be fused
        
        out_tokens = []
        out_sizes = []
        new_mappings = []
        
        # ------------------------------------------------------------------
        # Step 5: Weighted Token Aggregation & Spatial Provenance Update
        # ------------------------------------------------------------------
        for b in range(B):
            b_tokens_a = tokens_a[b]
            b_tokens_b = tokens_b[b]
            b_sizes_a = sizes_a[b]
            b_sizes_b = sizes_b[b]
            
            b_topk_a = topk_a_idx[b]          # Indices in partition A chosen for fusion
            b_topk_b = max_idx[b, b_topk_a]    # Matched destination indices in partition B
            
            # Mask to separate fused vs unfused tokens in partition A
            fused_a_mask = torch.zeros(b_tokens_a.shape[0], dtype=torch.bool, device=x.device)
            fused_a_mask[b_topk_a] = True
            
            # Unfused tokens from A stay as independent tokens
            unfused_a = b_tokens_a[~fused_a_mask]
            unfused_sizes_a = b_sizes_a[~fused_a_mask]
            
            # Tokens from A to be merged into B
            src_a = b_tokens_a[b_topk_a]
            src_sz = b_sizes_a[b_topk_a]
            dst_idx = b_topk_b
            
            # Size-weighted averaging:
            # new_b = (b * size_b + sum(src_a * size_a)) / (size_b + sum(size_a))
            weighted_b = b_tokens_b * b_sizes_b
            weighted_src_a = src_a * src_sz
            
            new_sizes_b = b_sizes_b.clone()
            new_sizes_b.scatter_add_(0, dst_idx.unsqueeze(1), src_sz)
            
            new_weighted_b = weighted_b.clone()
            dst_idx_feat = dst_idx.unsqueeze(1).expand(-1, C)
            new_weighted_b.scatter_add_(0, dst_idx_feat, weighted_src_a)
            
            final_b = new_weighted_b / new_sizes_b
            
            # Combine remaining tokens: [unfused_a, updated_b]
            b_out = torch.cat([unfused_a, final_b], dim=0)
            b_out_sz = torch.cat([unfused_sizes_a, new_sizes_b], dim=0)
            
            out_tokens.append(b_out)
            out_sizes.append(b_out_sz)
            
            # --------------------------------------------------------------
            # Step 6: Provenance Tracking (Mapping Old Indices -> New Indices)
            # This enables dense pixel-level anomaly heatmap reconstruction
            # after feature extraction without losing spatial defect resolution.
            # --------------------------------------------------------------
            b_mapping = mapping[b].clone()
            
            old_to_new = torch.zeros(N, dtype=torch.long, device=x.device)
            old_to_new[0] = 0  # [CLS] token always stays at index 0
            
            num_unfused = unfused_a.shape[0]
            
            # 1. B tokens map to: 1 + num_unfused + their index in B
            old_to_new[b_idx + 1] = 1 + num_unfused + torch.arange(b_idx.shape[0], device=x.device)
            
            # 2. Unfused A tokens map to: 1 + their new index in unfused_a
            unfused_a_orig_idx = a_idx[~fused_a_mask]
            old_to_new[unfused_a_orig_idx + 1] = 1 + torch.arange(num_unfused, device=x.device)
            
            # 3. Fused A tokens map to where their destination B token mapped
            fused_a_orig_idx = a_idx[b_topk_a]
            fused_a_dest_b_orig_idx = b_idx[b_topk_b]
            old_to_new[fused_a_orig_idx + 1] = old_to_new[fused_a_dest_b_orig_idx + 1]
            
            # Propagate through existing historical mapping chain
            b_new_mapping = old_to_new[b_mapping]
            new_mappings.append(b_new_mapping)
            
        out_tokens = torch.stack(out_tokens, dim=0)
        out_sizes = torch.stack(out_sizes, dim=0)
        final_mapping = torch.stack(new_mappings, dim=0)
        
        # Re-attach [CLS] token to the fused sequence
        final_out = torch.cat([cls_token, out_tokens], dim=1)
        final_sizes = torch.cat([cls_size, out_sizes], dim=1)
        
        return final_out, final_sizes, final_mapping
