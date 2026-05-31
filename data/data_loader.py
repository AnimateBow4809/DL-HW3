import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.mask_utils import rgb_to_class_index, DATASET1_COLOR_MAP, DATASET2_COLOR_MAP


class FootballDataset1(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        mask_idx = rgb_to_class_index(mask_rgb, DATASET1_COLOR_MAP)
        if self.transform:
            augmented = self.transform(image=image, mask=mask_idx)
            image = augmented['image']
            mask_idx = augmented['mask']
        return image, torch.as_tensor(mask_idx, dtype=torch.long)


class FootballDataset2(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = list(filter(lambda s: '.png' not in s, sorted(os.listdir(image_dir))))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = img_path + '___fuse.png'
        image = np.array(Image.open(img_path).convert("RGB"))
        mask_raw = np.array(Image.open(mask_path))
        if mask_raw.shape[-1] == 4:
            mask_rgb = mask_raw[:, :, :3]
        else:
            mask_rgb = mask_raw
        mask_idx = rgb_to_class_index(mask_rgb, DATASET2_COLOR_MAP)
        if self.transform:
            augmented = self.transform(image=image, mask=mask_idx)
            image = augmented['image']
            mask_idx = augmented['mask']

        return image, torch.as_tensor(mask_idx, dtype=torch.long)
