import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class MVTecDataset(Dataset):
    def __init__(self, root: str, category: str, is_train: bool = True, transform=None, mask_transform=None):
        """
        root: Path to mvtec_ad folder.
        category: MVTec category (e.g., 'bottle', 'hazelnut').
        is_train: True for training (normal only), False for testing (normal + anomaly).
        """
        self.dataset_path = os.path.join(root, category)
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path {self.dataset_path} not found. Please run scripts/download_dataset.py first.")
            
        self.is_train = is_train
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.img_paths = []
        self.labels = [] # 0 for normal, 1 for anomaly
        self.mask_paths = [] # None if normal
        
        if is_train:
            train_dir = os.path.join(self.dataset_path, "train", "good")
            if not os.path.exists(train_dir):
                raise FileNotFoundError(f"Train directory {train_dir} not found. Please run scripts/download_dataset.py first.")
            for img_name in sorted(os.listdir(train_dir)):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.img_paths.append(os.path.join(train_dir, img_name))
                    self.labels.append(0)
                    self.mask_paths.append(None)
        else:
            test_dir = os.path.join(self.dataset_path, "test")
            if not os.path.exists(test_dir):
                raise FileNotFoundError(f"Test directory {test_dir} not found. Please run scripts/download_dataset.py first.")
            
            ground_truth_dir = os.path.join(self.dataset_path, "ground_truth")
            
            for defect_type in sorted(os.listdir(test_dir)):
                defect_dir = os.path.join(test_dir, defect_type)
                if not os.path.isdir(defect_dir):
                    continue
                    
                label = 0 if defect_type == 'good' else 1
                
                for img_name in sorted(os.listdir(defect_dir)):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.img_paths.append(os.path.join(defect_dir, img_name))
                        self.labels.append(label)
                        
                        if label == 1:
                            mask_name = img_name.rsplit('.', 1)[0] + '_mask.png'
                            mask_path = os.path.join(ground_truth_dir, defect_type, mask_name)
                            if os.path.exists(mask_path):
                                self.mask_paths.append(mask_path)
                            else:
                                self.mask_paths.append(None)
                        else:
                            self.mask_paths.append(None)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
            
        mask = None
        if mask_path is not None:
            mask = Image.open(mask_path).convert('L')
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            else:
                mask = T.ToTensor()(mask)
        else:
            # dummy mask for normal
            if isinstance(img, torch.Tensor):
                mask = torch.zeros((1, img.shape[1], img.shape[2]), dtype=torch.float32)
            else:
                w, h = img.size
                mask = torch.zeros((1, h, w), dtype=torch.float32)
                
        return {
            'image': img,
            'label': label,
            'mask': mask
        }

def get_transforms(img_size: int = 224):
    """
    Standard ViT transforms.
    """
    transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor()
    ])
    
    return transform, mask_transform
