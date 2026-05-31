import torch
import numpy as np
from numpy import ndarray
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary

from utils.visualization import plot_learning_curves, plot_example
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassConfusionMatrix

class Trainer:
    def __init__(self, model, optimizer, loss_fn, num_classes, device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.device = device if device else next(model.parameters()).device
        self.model.to(self.device)
        self.iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(self.device)
        self.conf_matrix_metric = MulticlassConfusionMatrix(num_classes=num_classes).to(self.device)

        self.history = {
            'train_loss': [], 'train_iou': [],
            'val_loss': [], 'val_iou': []
        }
        summary(self.model)

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs=50,
              use_early_stopping: bool = False, patience: int = 3, min_delta: float = 1e-4):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0.0
            self.iou_metric.reset()

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
            for x_batch, y_batch in train_pbar:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                output = self.model(x_batch)
                loss = self.loss_fn(output, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item() * x_batch.size(0)
                preds = torch.argmax(output, dim=1)

                self.iou_metric.update(preds, y_batch)

            avg_train_loss = epoch_train_loss / len(train_loader.dataset)
            avg_train_iou = self.iou_metric.compute().item()

            self.history['train_loss'].append(avg_train_loss)
            self.history['train_iou'].append(avg_train_iou)

            if val_loader:
                val_loss, val_iou = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_iou'].append(val_iou)

                print(
                    f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f} | Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

                if use_early_stopping:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            print(f"\nEarly stopping triggered at epoch {epoch + 1}. Training halted.")
                            break
            else:
                print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}")

    def evaluate(self, dataLoader: DataLoader):
        total_loss = 0.0
        self.model.eval()
        self.iou_metric.reset()

        eval_pbar = tqdm(dataLoader, desc="[Evaluate]", leave=False)
        with torch.no_grad():
            for x_batch, y_batch in eval_pbar:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                output = self.model(x_batch)
                loss = self.loss_fn(output, y_batch)
                total_loss += loss.item() * x_batch.size(0)

                preds = torch.argmax(output, dim=1)
                self.iou_metric.update(preds, y_batch)

        avg_loss = total_loss / len(dataLoader.dataset)
        avg_iou = self.iou_metric.compute().item()

        return avg_loss, avg_iou

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.device)
        print(f"Model loaded from {filepath}")

    def get_per_class_metrics(self, dataLoader: DataLoader):
        self.model.eval()
        self.conf_matrix_metric.reset()
        eval_pbar = tqdm(dataLoader, desc="[Computing Confusion Matrix]", leave=False)
        with torch.no_grad():
            for x_batch, y_batch in eval_pbar:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                output = self.model(x_batch)
                preds = torch.argmax(output, dim=1)
                self.conf_matrix_metric.update(preds, y_batch)
        conf_matrix = self.conf_matrix_metric.compute().cpu().numpy()
        per_class_stats = {}
        for i in range(self.num_classes):
            tp = conf_matrix[i, i]
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp
            denominator = tp + fp + fn
            iou = tp / denominator if denominator > 0 else 0.0
            per_class_stats[i] = {
                "IoU": iou,
                "total_pixels": conf_matrix[i, :].sum()
            }

        return per_class_stats

    def plot_learning_curves(self):
        plot_learning_curves(self.history)

    def plot_predictions(self, dataloader, n=5):
        self.model.eval()
        samples_plotted = 0
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                outputs = self.model(x_batch)
                preds = torch.argmax(outputs, dim=1)
                x_batch = x_batch.cpu().numpy()
                y_batch = y_batch.cpu().numpy()
                preds = preds.cpu().numpy()
                for i in range(x_batch.shape[0]):
                    if samples_plotted >= n:
                        return
                    plot_example(x_batch[i], y_batch[i], preds[i])
                    samples_plotted += 1
