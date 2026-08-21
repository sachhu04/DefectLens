import torch
import numpy as np
import os
import json

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors

class PatchCoreAnomalyDetector:
    def __init__(self, coreset_sampling_ratio: float = 0.01, num_neighbors: int = 9):
        """
        coreset_sampling_ratio: Proportion of normal features to retain in memory bank.
        """
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.memory_bank = None
        self.faiss_index = None
        self.knn = None

    def fit(self, features_list: list):
        """
        features_list: List of tensors of shape [N, C] where N is number of patches.
        """
        # Concatenate all features
        all_features = torch.cat(features_list, dim=0) # [Total_N, C]
        
        # Coreset Subsampling (random sampling for simplicity in this version,
        # PatchCore typically uses k-Center Greedy, but random works as a strong baseline
        # when efficiency is prioritized, or when N is manageable).
        num_samples = max(1, int(all_features.shape[0] * self.coreset_sampling_ratio))
        
        # Random subsampling
        indices = torch.randperm(all_features.shape[0])[:num_samples]
        self.memory_bank = all_features[indices].cpu().numpy()
        
        self._build_index()

    def _build_index(self):
        if self.memory_bank is None:
            return
            
        dim = self.memory_bank.shape[1]
        
        if HAS_FAISS:
            self.faiss_index = faiss.IndexFlatL2(dim)
            self.faiss_index.add(self.memory_bank)
        else:
            self.knn = NearestNeighbors(n_neighbors=self.num_neighbors, metric='euclidean', n_jobs=-1)
            self.knn.fit(self.memory_bank)

    def predict(self, features: torch.Tensor, mapping: torch.Tensor = None, patch_grid_size: int = 14) -> tuple:
        """
        features: [B, N_reduced, C] - Reduced patch features (without CLS token).
        mapping: [B, N_orig] - Mapping from original patch to reduced token index.
        patch_grid_size: H=W of the original patch grid (e.g., 14 for 224//16).
        
        Returns:
            anomaly_map: [B, patch_grid_size, patch_grid_size]
            image_score: [B]
        """
        B, N_reduced, C = features.shape
        features_np = features.detach().cpu().numpy().reshape(-1, C)
        
        if HAS_FAISS:
            distances, _ = self.faiss_index.search(features_np, self.num_neighbors)
        else:
            distances, _ = self.knn.kneighbors(features_np)
            
        # PatchCore score is the distance to the nearest neighbor, optionally re-weighted
        # by the distance to the k-th nearest neighbor to increase robustness.
        # Here we use the distance to the closest neighbor for simplicity.
        min_distances = distances[:, 0].reshape(B, N_reduced)
        
        # distances to tensor
        min_distances = torch.from_numpy(min_distances).to(features.device)
        
        # Now we need to map these reduced scores back to the original 2D grid.
        # min_distances: [B, N_reduced]
        # mapping: [B, N_orig]
        
        if mapping is not None:
            N_orig = mapping.shape[1]
            # anomaly_scores_orig[b, i] = min_distances[b, mapping[b, i]]
            # mapping contains indices in the range [0, N_reduced-1]
            anomaly_scores_orig = torch.gather(min_distances, 1, mapping)
        else:
            # If no mapping, assume N_reduced == N_orig
            anomaly_scores_orig = min_distances
            
        # Reshape to 2D grid
        anomaly_map = anomaly_scores_orig.reshape(B, patch_grid_size, patch_grid_size)
        
        # Image-level score (max patch score)
        image_score, _ = anomaly_scores_orig.max(dim=1)
        
        return anomaly_map, image_score

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        if self.memory_bank is not None:
            np.save(os.path.join(save_dir, "embeddings.npy"), self.memory_bank)
            
        metadata = {
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "num_neighbors": self.num_neighbors,
            "dim": self.memory_bank.shape[1] if self.memory_bank is not None else 0
        }
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def load(self, load_dir: str):
        emb_path = os.path.join(load_dir, "embeddings.npy")
        if os.path.exists(emb_path):
            self.memory_bank = np.load(emb_path)
            
        meta_path = os.path.join(load_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
                self.coreset_sampling_ratio = metadata.get("coreset_sampling_ratio", 0.01)
                self.num_neighbors = metadata.get("num_neighbors", 9)
                
        self._build_index()
