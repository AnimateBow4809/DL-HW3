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

    ax2.plot(epochs, clean_metric(history['train_acc']), label='Train Accuracy', marker='o')
    if history.get('val_acc'):
        ax2.plot(epochs, clean_metric(history['val_acc']), label='Validation Accuracy', marker='o')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)

    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
