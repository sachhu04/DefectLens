"""MVTec AD dataset loading and transform helpers."""

import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


def get_transforms(image_size: int = 224) -> tuple:
    image_transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    mask_transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])
    return image_transform, mask_transform


class MVTecDataset(Dataset):

    def __init__(self, root: str, category: str, is_train: bool = True,
                 transform=None, mask_transform=None):
        self.root = root
        self.category = category
        self.is_train = is_train
        self.transform = transform
        self.mask_transform = mask_transform

        split = 'train' if is_train else 'test'
        self.split_dir = os.path.join(root, category, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(
                f"MVTec split not found: {self.split_dir}. "
                f"Run scripts/download_dataset.py or place the MVTec AD data here.")

        self.samples = []  # (image_path, label)
        for object_name in sorted(os.listdir(self.split_dir)):
            obj_dir = os.path.join(self.split_dir, object_name)
            if not os.path.isdir(obj_dir):
                continue
            label = 0 if object_name == 'good' else 1
            for img_name in sorted(os.listdir(obj_dir)):
                if img_name.lower().endswith(_IMAGE_EXTS):
                    self.samples.append((os.path.join(obj_dir, img_name), label))

        if not self.samples:
            raise FileNotFoundError(f"No images found under {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        if self.is_train:
            # Train split has no defects and no ground truth.
            mask = torch.zeros(1, image.shape[-2], image.shape[-1])
        else:
            mask = self._load_mask(img_path)
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)

        return {'image': image, 'label': label, 'mask': mask}

    def _load_mask(self, img_path: str) -> Image.Image:
        # data/<category>/ground_truth/<defect_type>/<image>_mask.png
        rel = os.path.relpath(img_path, self.split_dir)  # e.g. scratch/x_test_1.png
        defect_type, img_name = os.path.split(rel)
        mask_name = os.path.splitext(img_name)[0] + '_mask.png'
        mask_path = os.path.join(self.root, self.category, 'ground_truth', defect_type, mask_name)
        if os.path.exists(mask_path):
            return Image.open(mask_path).convert('L')
        # 'good' images carry no ground truth.
        return Image.new('L', Image.open(img_path).size, 0)
