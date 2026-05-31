import random

import numpy as np
import yaml
import torch
import os
from torch.utils.data import ConcatDataset, random_split
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

def get_datasets(config):
    data_cfg = config.get("data", {})
    dataset1_path = data_cfg.get("dataset1_path", '../data/dataset/dataset1')
    dataset2_path = data_cfg.get("dataset2_path", '../data/dataset/dataset2')

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    ds1_img_dir = os.path.join(dataset1_path, "images")
    ds1_mask_dir = os.path.join(dataset1_path, "masks")
    dataset1 = FootballDataset1(image_dir=ds1_img_dir, mask_dir=ds1_mask_dir, transform=transform)

    ds2_img_dir = os.path.join(dataset2_path, "images")
    dataset2 = FootballDataset2(image_dir=ds2_img_dir, transform=transform)

    def split_dataset(ds, train_pct=0.8, val_pct=0.1):
        total_len = len(ds)
        train_len = int(total_len * train_pct)
        val_len = int(total_len * val_pct)
        test_len = total_len - train_len - val_len
        generator = torch.Generator().manual_seed(42)
        return random_split(ds, [train_len, val_len, test_len], generator=generator)


    train1, val1, test1 = split_dataset(dataset1)
    train2, val2, test2 = split_dataset(dataset2)

    train_dataset = ConcatDataset([train1, train2])
    val_dataset = ConcatDataset([val1, val2])
    test_dataset = ConcatDataset([test1, test2])

    print(f"Train size: {len(train_dataset)} (DS1: {len(train1)}, DS2: {len(train2)})")
    print(f"Val size: {len(val_dataset)} (DS1: {len(val1)}, DS2: {len(val2)})")
    print(f"Test size: {len(test_dataset)} (DS1: {len(test1)}, DS2: {len(test2)})")

    return train_dataset, val_dataset, test_dataset
