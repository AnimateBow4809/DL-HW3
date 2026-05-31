import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.mask_utils import CLASS_NAMES

plt.style.use('default')

def analyze_class_distribution(dataloader, num_classes=9):
    total_pixels = torch.zeros(num_classes, dtype=torch.long)
    print(f"Analyzing {len(dataloader.dataset)} images...")
    for images, masks in tqdm(dataloader, desc="Counting Pixels"):
        counts = torch.bincount(masks.flatten(), minlength=num_classes)
        total_pixels += counts.cpu()

    total_sum = total_pixels.sum().float()
    percentages = (total_pixels.float() / total_sum) * 100

    print("\n--- Class Distribution ---")
    for i in range(num_classes):
        print(f"{CLASS_NAMES[i]:<20}: {percentages[i]:>8.4f}%  ({total_pixels[i]:,} pixels)")

    plt.figure(figsize=(12, 6))
    bars = plt.bar(CLASS_NAMES, percentages.numpy(), color='royalblue', edgecolor='black')
    plt.yscale('log')
    plt.title('Pixel Distribution per Class (Log Scale)', fontsize=16)
    plt.ylabel('Percentage of Total Pixels (Log %)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, pct in zip(bars, percentages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval * 1.2, f'{pct:.2f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    return total_pixels, percentages

def plot_learning_curves(history):
    # Ensure any stray PyTorch tensors in history (e.g. from loss.item()) are floats
    def clean_metric(metric_list):
        if not metric_list: return metric_list
        return [m.item() if torch.is_tensor(m) else m for m in metric_list]

    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, clean_metric(history['train_loss']), label='Train Loss', marker='o')
    if history.get('val_loss'):
        ax1.plot(epochs, clean_metric(history['val_loss']), label='Validation Loss', marker='o')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, clean_metric(history['train_iou']), label='Train IOU', marker='o')
    if history.get('val_iou'):
        ax2.plot(epochs, clean_metric(history['val_iou']), label='Validation IOU', marker='o')
    ax2.set_title('IOU over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IOU')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_example(x_sample: np.ndarray, y_sample: np.ndarray, pred_sample: np.ndarray):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    num_classes = 8
    img = np.transpose(x_sample, (1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(y_sample, cmap='tab10', vmin=0, vmax=num_classes - 1)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    axes[2].imshow(pred_sample, cmap='tab10', vmin=0, vmax=num_classes - 1)
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()