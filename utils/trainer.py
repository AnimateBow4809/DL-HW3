import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary

from utils.visualization import plot_learning_curves, plot_confusion_matrix, plot_image_predictions, plot_feature_maps


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device if device else next(model.parameters()).device
        self.model.to(self.device)
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        summary(self.model)

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs=50,
              use_early_stopping: bool = False, patience: int = 3, min_delta: float = 1e-4):
        num_train_samples = len(train_loader.dataset)
        best_val_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0.0
            epoch_train_correct = 0
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
                epoch_train_correct += (preds == y_batch).sum().item()
            avg_train_loss = epoch_train_loss / num_train_samples
            avg_train_acc = epoch_train_correct / num_train_samples
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(avg_train_acc)

            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                print(
                    f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                if use_early_stopping:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        print(f"EarlyStopping counter: {epochs_no_improve} out of {patience}")
                        if epochs_no_improve >= patience:
                            print(f"\nEarly stopping triggered at epoch {epoch + 1}. Training halted.")
                            break
            else:
                print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
                if use_early_stopping:
                    print(
                        "Warning: Early stopping is enabled but no validation loader was provided. It will be ignored.")

    def evaluate(self, dataLoader: DataLoader):
        num_samples = len(dataLoader.dataset)
        total_loss = 0.0
        total_correct = 0

        self.model.eval()

        eval_pbar = tqdm(dataLoader, desc="[Evaluate]", leave=False)
        with torch.no_grad():
            for x_batch, y_batch in eval_pbar:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                output = self.model(x_batch)
                loss = self.loss_fn(output, y_batch)
                total_loss += loss.item() * x_batch.size(0)
                preds = torch.argmax(output, dim=1)
                total_correct += (preds == y_batch).sum().item()

        return total_loss / num_samples, total_correct / num_samples

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.device)
        print(f"Model loaded from {filepath}")

    def plot_misclassified_predictions(self, dataLoader: DataLoader, num_images: int = 5,
                                       image_shape: tuple = (28, 28)):
        misclassified_imgs = []
        misclassified_trues = []
        misclassified_preds = []
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in dataLoader:
                x_cpu, y_cpu = x_batch.cpu(), y_batch.cpu()
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                output = self.model(x_batch)
                preds = torch.argmax(output, dim=1)
                wrong_indices = (preds != y_batch).nonzero(as_tuple=True)[0]
                for idx in wrong_indices:
                    misclassified_imgs.append(x_cpu[idx].numpy())
                    misclassified_trues.append(y_cpu[idx].item())
                    misclassified_preds.append(preds[idx].item())

                    if len(misclassified_imgs) >= num_images:
                        break

                if len(misclassified_imgs) >= num_images:
                    break

        if len(misclassified_imgs) == 0:
            print("No misclassified images found! The model is 100% accurate on this data.")
            return

        plot_image_predictions(misclassified_imgs, misclassified_trues, misclassified_preds, image_shape)

    def get_per_class_accuracy(self, dataLoader: DataLoader, num_classes: int = 10):
        all_labels, all_preds = self._predict(dataLoader)
        per_class_accuracy = {}
        for i in range(num_classes):
            class_mask = (all_labels == i)
            total_samples = np.sum(class_mask)
            if total_samples > 0:
                correct_predictions = np.sum((all_preds == i) & class_mask)
                accuracy = correct_predictions / total_samples
            else:
                accuracy = 0.0
            per_class_accuracy[i] = {"accuracy": accuracy, "total_samples": total_samples}
        return per_class_accuracy

    def plot_learning_curves(self):
        plot_learning_curves(self.history)

    def plot_confusion_matrix(self, dataLoader: DataLoader, num_classes: int = 10):
        y_true, y_pred = self._predict(dataLoader)
        plot_confusion_matrix(y_true, y_pred, class_names=[i for i in range(num_classes)])

    def plot_feature_maps(self, dataLoader: DataLoader):
        input = next(iter(dataLoader))[0][0].unsqueeze(0)
        x, y, z = get_first_block_activation(self.model, input)
        plot_feature_maps(input,x,y,z)
    def _predict(self, dataLoader: DataLoader):
        all_preds = []
        all_labels = []

        self.model.eval()
        eval_pbar = tqdm(dataLoader, desc="[Predicting]", leave=False)

        with torch.no_grad():
            for x_batch, y_batch in eval_pbar:
                x_batch = x_batch.to(self.device)
                output = self.model(x_batch)
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())

        return np.array(all_labels), np.array(all_preds)
