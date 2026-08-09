import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MCTFModule(nn.Module):
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
        if not self.use_info or next_block is None:
            return torch.ones((x.shape[0], x.shape[1], 1), device=x.device, dtype=x.dtype)
            
        B, N, C = x.shape
        attn = next_block.attn
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_scores = (q @ k.transpose(-2, -1)) * attn.scale
        attn_probs = attn_scores.softmax(dim=-1)
        
        info = attn_probs.mean(dim=1).sum(dim=1, keepdim=True)
        info = info.transpose(1, 2)
        info = info / (info.max(dim=1, keepdim=True)[0] + 1e-8)
        return info

    def forward(self, x: torch.Tensor, token_sizes: torch.Tensor, mapping: torch.Tensor = None, next_block: nn.Module = None) -> tuple:
        B, N, C = x.shape
        num_reduce = int((N - 1) * self.r)
        
        if mapping is None:
            mapping = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1).clone()

        if num_reduce <= 0:
            return x, token_sizes, mapping
            
        cls_token = x[:, 0:1, :]
        cls_size = token_sizes[:, 0:1, :]
        
        tokens = x[:, 1:, :]
        sizes = token_sizes[:, 1:, :]
        num_patches = tokens.shape[1]
        
        a_idx = torch.arange(0, num_patches, 2, device=x.device)
        b_idx = torch.arange(1, num_patches, 2, device=x.device)
        
        tokens_a = tokens[:, a_idx, :]
        tokens_b = tokens[:, b_idx, :]
        sizes_a = sizes[:, a_idx, :]
        sizes_b = sizes[:, b_idx, :]
        
        if self.use_sim:
            norm_a = F.normalize(tokens_a, dim=-1)
            norm_b = F.normalize(tokens_b, dim=-1)
            sim = (norm_a @ norm_b.transpose(-2, -1) + 1.0) / 2.0 
            W_sim = sim ** self.tau_sim
        else:
            W_sim = 1.0
            
        if self.use_info:
            info = self.get_informativeness(x, next_block)
            info_a = info[:, 1:, :][:, a_idx, :]
            info_b = info[:, 1:, :][:, b_idx, :]
            info_sum = info_a + info_b.transpose(1, 2)
            W_info = torch.exp(-info_sum) ** self.tau_info
        else:
            W_info = 1.0
            
        if self.use_size:
            size_sum = sizes_a + sizes_b.transpose(1, 2)
            W_size = (1.0 / size_sum) ** self.tau_size
        else:
            W_size = 1.0
            
        W = W_sim * W_info * W_size
        max_W, max_idx = W.max(dim=-1)
        topk_W, topk_a_idx = max_W.topk(num_reduce, dim=-1)
        
        out_tokens = []
        out_sizes = []
        new_mappings = []
        
        for b in range(B):
            b_tokens_a = tokens_a[b]
            b_tokens_b = tokens_b[b]
            b_sizes_a = sizes_a[b]
            b_sizes_b = sizes_b[b]
            
            b_topk_a = topk_a_idx[b]
            b_topk_b = max_idx[b, b_topk_a]
            
            fused_a_mask = torch.zeros(b_tokens_a.shape[0], dtype=torch.bool, device=x.device)
            fused_a_mask[b_topk_a] = True
            
            unfused_a = b_tokens_a[~fused_a_mask]
            unfused_sizes_a = b_sizes_a[~fused_a_mask]
            
            src_a = b_tokens_a[b_topk_a]
            src_sz = b_sizes_a[b_topk_a]
            dst_idx = b_topk_b
            
            weighted_b = b_tokens_b * b_sizes_b
            weighted_src_a = src_a * src_sz
            
            new_sizes_b = b_sizes_b.clone()
            new_sizes_b.scatter_add_(0, dst_idx.unsqueeze(1), src_sz)
            
            new_weighted_b = weighted_b.clone()
            dst_idx_feat = dst_idx.unsqueeze(1).expand(-1, C)
            new_weighted_b.scatter_add_(0, dst_idx_feat, weighted_src_a)
            
            final_b = new_weighted_b / new_sizes_b
            
            b_out = torch.cat([unfused_a, final_b], dim=0)
            b_out_sz = torch.cat([unfused_sizes_a, new_sizes_b], dim=0)
            
            out_tokens.append(b_out)
            out_sizes.append(b_out_sz)
            
            # Update mapping for this batch item
            b_mapping = mapping[b].clone()
            
            # Map old indices to new indices
            # old A indices are a_idx + 1
            # old B indices are b_idx + 1
            old_to_new = torch.zeros(N, dtype=torch.long, device=x.device)
            old_to_new[0] = 0 # CLS remains 0
            
            num_unfused = unfused_a.shape[0]
            
            # 1. B tokens map to 1 + num_unfused + their index in B
            old_to_new[b_idx + 1] = 1 + num_unfused + torch.arange(b_idx.shape[0], device=x.device)
            
            # 2. Unfused A tokens map to 1 + their new index in unfused_a
            unfused_a_orig_idx = a_idx[~fused_a_mask]
            old_to_new[unfused_a_orig_idx + 1] = 1 + torch.arange(num_unfused, device=x.device)
            
            # 3. Fused A tokens map to where their destination B token maps
            fused_a_orig_idx = a_idx[b_topk_a]
            fused_a_dest_b_orig_idx = b_idx[b_topk_b]
            old_to_new[fused_a_orig_idx + 1] = old_to_new[fused_a_dest_b_orig_idx + 1]
            
            # Apply to b_mapping
            b_new_mapping = old_to_new[b_mapping]
            new_mappings.append(b_new_mapping)
            
        out_tokens = torch.stack(out_tokens, dim=0)
        out_sizes = torch.stack(out_sizes, dim=0)
        final_mapping = torch.stack(new_mappings, dim=0)
        
        final_out = torch.cat([cls_token, out_tokens], dim=1)
        final_sizes = torch.cat([cls_size, out_sizes], dim=1)
        
        return final_out, final_sizes, final_mapping
