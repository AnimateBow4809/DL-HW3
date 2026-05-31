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

