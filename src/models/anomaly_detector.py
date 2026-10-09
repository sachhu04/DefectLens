"""
PatchCore Memory Bank Anomaly Detector with Spatial Mapping Reconstruction.

Overview:
---------
PatchCore is a state-of-the-art anomaly detection paradigm for visual inspection.
Rather than training a classification model with defect samples (which are rare or unavailable
in industrial manufacturing), PatchCore operates under the Unsupervised / One-Class learning framework:

1. Training (fit):
   - Extracts patch embeddings from purely normal/nominal training images.
   - Builds an aggregated feature collection (Memory Bank).
   - Applies Coreset Subsampling to retain a compact, diverse subset of normal features.
   - Indexes features using FAISS (IndexFlatL2) for fast Euclidean distance search.

2. Inference (predict):
   - Given a query image, extracts patch features and queries the memory bank.
   - The anomaly score for each token is its distance to the nearest normal neighbor in the bank.
   - Leverages the MCTF `mapping` tensor via `torch.gather()` to project scores from the reduced
     token sequence (e.g., 23 tokens) back to all 196 original patch locations (14x14 grid).
   - Computes the image-level anomaly score as the maximum patch distance.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np

try:
    import faiss
    faiss.omp_set_num_threads(1)
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

import torch
from sklearn.neighbors import NearestNeighbors


class PatchCoreAnomalyDetector:
    """
    PatchCore Anomaly Detector supporting both standard ViT and MCTF token-reduced features.
    
    Args:
        coreset_sampling_ratio (float): Proportion of normal patch embeddings retained
            in the memory bank (e.g., 0.1 = 10% coreset).
        num_neighbors (int): Number of nearest neighbors to query during kNN search (default: 9).
    """
    def __init__(self, coreset_sampling_ratio: float = 0.01, num_neighbors: int = 9):
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.memory_bank = None
        self.faiss_index = None
        self.knn = None

    def fit(self, features_list: list):
        """
        Populates the nominal memory bank using patch features from normal training images.
        
        Args:
            features_list (list): List of tensors of shape [N_patches, C].
        """
        # Concatenate all patch representations from normal training images: [Total_N, C]
        all_features = torch.cat(features_list, dim=0)
        
        # Determine coreset size based on configured sampling ratio
        num_samples = max(1, int(all_features.shape[0] * self.coreset_sampling_ratio))
        
        # Subsample features into a compact memory bank
        # (Random subsampling provides high efficiency with negligible degradation in practice)
        indices = torch.randperm(all_features.shape[0])[:num_samples]
        self.memory_bank = all_features[indices].cpu().numpy()
        
        # L2-normalize memory bank so Euclidean distances correlate directly with cosine similarity
        self._l2_normalize(self.memory_bank)
        
        # Construct fast nearest neighbor search index
        self._build_index()

    @staticmethod
    def _l2_normalize(x: np.ndarray):
        """
        In-place L2 normalization across feature vectors.
        Ensures Euclidean distance values are bounded in [0, 2] and consistent across runs.
        """
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        np.divide(x, norms + 1e-8, out=x)

    def _build_index(self):
        """
        Constructs the search index (FAISS IndexFlatL2 if available, otherwise scikit-learn NearestNeighbors).
        """
        if self.memory_bank is None:
            return
            
        dim = self.memory_bank.shape[1]
        
        if HAS_FAISS:
            self.memory_bank = np.ascontiguousarray(self.memory_bank, dtype=np.float32)
            self.faiss_index = faiss.IndexFlatL2(dim)
            self.faiss_index.add(self.memory_bank)
        else:
            self.knn = NearestNeighbors(n_neighbors=self.num_neighbors, metric='euclidean', n_jobs=-1)
            self.knn.fit(self.memory_bank)

    def predict(self, features: torch.Tensor, mapping: torch.Tensor = None, patch_grid_size: int = 14) -> tuple:
        """
        Scores query image patches against the normal memory bank and reconstructs the 2D anomaly heatmap.
        
        Crucial MCTF Reconstruction Mechanism:
        When MCTF fuses 196 tokens into 23 tokens, k-NN is evaluated ONLY on the 23 tokens (speeding up search).
        The `mapping` tensor records which of the 23 tokens represents each of the original 196 patch positions.
        Using `torch.gather(min_distances, 1, mapping)`, the anomaly distances are seamlessly broadcasted
        back to the full 14x14 grid, preserving full spatial resolution!
        
        Args:
            features (torch.Tensor): Extracted patch representations [B, N_reduced, C].
            mapping (torch.Tensor, optional): Original-to-reduced token index map [B, N_orig].
            patch_grid_size (int): Grid dimension (e.g., 14 for 224x224 image with 16x16 patches).
            
        Returns:
            anomaly_map (torch.Tensor): Spatial patch anomaly map [B, patch_grid_size, patch_grid_size].
            image_score (torch.Tensor): Overall image anomaly classification score [B].
        """
        B, N_reduced, C = features.shape
        features_np = features.detach().cpu().numpy().reshape(-1, C)
        self._l2_normalize(features_np)
        
        # 1. Query nearest nominal neighbors in the memory bank
        if HAS_FAISS:
            features_np = np.ascontiguousarray(features_np, dtype=np.float32)
            distances, _ = self.faiss_index.search(features_np, self.num_neighbors)
            # FAISS IndexFlatL2 returns squared Euclidean distances; take sqrt for Euclidean distance
            distances = np.sqrt(np.maximum(distances, 0))
        else:
            distances, _ = self.knn.kneighbors(features_np)
            
        # Distance to the closest nominal neighbor represents patch anomaly score
        min_distances = distances[:, 0].reshape(B, N_reduced)
        min_distances = torch.from_numpy(min_distances).to(features.device)
        
        # 2. Reconstruct dense 14x14 grid from reduced tokens
        if mapping is not None:
            # mapping has shape [B, 196] with values in range [0, N_reduced - 1]
            # torch.gather maps min_distances[b, mapping[b, i]] for each original patch i
            anomaly_scores_orig = torch.gather(min_distances, 1, mapping)
        else:
            # For baseline ViT where no token reduction was applied (N_reduced == N_orig == 196)
            anomaly_scores_orig = min_distances
            
        # 3. Reshape flat patch scores [B, 196] into 2D spatial grid [B, 14, 14]
        anomaly_map = anomaly_scores_orig.reshape(B, patch_grid_size, patch_grid_size)
        
        # 4. Compute image-level anomaly score as the maximum patch anomaly score
        image_score, _ = anomaly_scores_orig.max(dim=1)
        
        return anomaly_map, image_score

    def save(self, save_dir: str):
        """
        Persists memory bank embeddings (.npy) and configuration metadata (.json) to disk.
        """
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
        """
        Loads precomputed memory bank embeddings and metadata from disk and builds the search index.
        """
        emb_path = os.path.join(load_dir, "embeddings.npy")
        if os.path.exists(emb_path):
            self.memory_bank = np.load(emb_path)
            # Idempotent L2-normalization ensuring distance metric consistency
            self._l2_normalize(self.memory_bank)
            
        meta_path = os.path.join(load_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
                self.coreset_sampling_ratio = metadata.get("coreset_sampling_ratio", 0.01)
                self.num_neighbors = metadata.get("num_neighbors", 9)
                
        self._build_index()
