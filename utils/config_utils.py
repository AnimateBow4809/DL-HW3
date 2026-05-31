import random

import numpy as np
import yaml
import torch
import os
from torch.utils.data import ConcatDataset, random_split, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.data_loader import FootballDataset1, FootballDataset2
from models.unet import UNet


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}. Deterministic algorithms enabled.")


def get_model(config):
    model_cfg = config.get("model", {})
    use_skip = model_cfg.get("skip_connection", True)
    expansion = model_cfg.get("expansion", "up")
    use_bn = model_cfg.get("batch_normalization", False)
    num_classes = model_cfg.get("num_classes", 8)
    device = model_cfg.get("device", "cpu")
    model_path = model_cfg.get("path", None)
    model = UNet(
        in_channels=3,
        out_channels=num_classes,
        use_bn=use_bn,
        expansion=expansion,
        use_skip=use_skip
    )
    if model_path is not None:
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded model weights from: {model_path}")
        except Exception as e:
            print(f"Error loading model weights from {model_path}: {e}")
    model = model.to(device)
    return model


def get_transforms():
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),  # Handles day/night and stadium shadows
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform


def get_datasets(config):
    data_cfg = config.get("data", {})
    dataset1_path = data_cfg.get("dataset1_path", '../data/dataset/dataset1')
    dataset2_path = data_cfg.get("dataset2_path", '../data/dataset/dataset2')

    train_transform, val_transform = get_transforms()
    ds1_img_dir = os.path.join(dataset1_path, "images")
    ds1_mask_dir = os.path.join(dataset1_path, "masks")

    ds1_train_base = FootballDataset1(image_dir=ds1_img_dir, mask_dir=ds1_mask_dir, transform=train_transform)
    ds1_val_base = FootballDataset1(image_dir=ds1_img_dir, mask_dir=ds1_mask_dir, transform=val_transform)

    ds2_img_dir = os.path.join(dataset2_path, "images")

    ds2_train_base = FootballDataset2(image_dir=ds2_img_dir, transform=train_transform)
    ds2_val_base = FootballDataset2(image_dir=ds2_img_dir, transform=val_transform)

    def get_split_indices(total_len, train_pct=0.8, val_pct=0.1):
        train_len = int(total_len * train_pct)
        val_len = int(total_len * val_pct)
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_len, generator=generator).tolist()
        train_idx = indices[:train_len]
        val_idx = indices[train_len:train_len + val_len]
        test_idx = indices[train_len + val_len:]
        return train_idx, val_idx, test_idx

    t_idx1, v_idx1, test_idx1 = get_split_indices(len(ds1_train_base))
    train1 = Subset(ds1_train_base, t_idx1)
    val1 = Subset(ds1_val_base, v_idx1)
    test1 = Subset(ds1_val_base, test_idx1)

    t_idx2, v_idx2, test_idx2 = get_split_indices(len(ds2_train_base))
    train2 = Subset(ds2_train_base, t_idx2)
    val2 = Subset(ds2_val_base, v_idx2)
    test2 = Subset(ds2_val_base, test_idx2)

    train_dataset = ConcatDataset([train1, train2])
    val_dataset = ConcatDataset([val1, val2])
    test_dataset = ConcatDataset([test1, test2])

    print(f"Train size: {len(train_dataset)} (DS1: {len(train1)}, DS2: {len(train2)}) | Augmented")
    print(f"Val size: {len(val_dataset)} (DS1: {len(val1)}, DS2: {len(val2)}) | Pristine")
    print(f"Test size: {len(test_dataset)} (DS1: {len(test1)}, DS2: {len(test2)}) | Pristine")

    return train_dataset, val_dataset, test_dataset
