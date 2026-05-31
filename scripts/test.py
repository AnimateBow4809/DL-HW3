import torch
import yaml
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from utils.config_utils import load_config, get_datasets, get_model, set_seed
from utils.trainer import Trainer

if __name__ == '__main__':
    set_seed()
    config = load_config('../config/config.yaml')
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})

    device = model_cfg.get('device', 'cpu')
    num_classes = model_cfg.get('num_classes', 8)
    batch_size = data_cfg.get('batch_size', 256)

    _, _, test_dataset = get_datasets(config)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = get_model(config)
    dummy_optimizer = torch.optim.Adam(model.parameters())

    trainer = Trainer(
        device=device,
        model=model,
        optimizer=dummy_optimizer,
        loss_fn=CrossEntropyLoss(),
        num_classes=num_classes,
    )

    print("\n--- Starting Evaluation on Test Set ---")
    test_loss, test_iou = trainer.evaluate(test_loader)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Overall IoU: {test_iou:.4f}\n")

    per_class_metrics = trainer.get_per_class_metrics(test_loader)
    print("--- Per-Class IoU ---")
    for i in range(num_classes):
        iou = per_class_metrics[i]['IoU']
        total_pixels = per_class_metrics[i]['total_pixels']
        print(f"Class {i} IoU: {iou:.4f} | Total Pixels in Test Set: {total_pixels}")