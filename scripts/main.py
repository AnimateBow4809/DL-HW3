import torch
import yaml
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from utils.focal_loss import FocalLoss
from utils.mask_utils import compute_class_weights
from utils.trainer import Trainer
from utils.config_utils import load_config, get_datasets, get_model, set_seed

if __name__ == '__main__':
    set_seed()
    config = load_config('../config/config.yaml')
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    opt_cfg = config.get('optimizer', {})
    use_focal_loss = model_cfg.get("use_focal_loss", False)
    device = model_cfg.get('device', 'cpu')
    epochs = model_cfg.get('epochs', 5)
    early_stopping = model_cfg.get('early_stopping', False)
    num_classes = model_cfg.get('num_classes', 8)
    batch_size = data_cfg.get('batch_size', 256)
    use_skip = model_cfg.get("skip_connection", True)
    expansion = model_cfg.get("expansion", "up")
    use_bn = model_cfg.get("batch_normalization", False)

    lr = opt_cfg.get('lr', 0.01)
    weight_decay = opt_cfg.get('weight_decay', 0.0001)
    train_dataset, val_dataset, test_dataset = get_datasets(config)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = get_model(config)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay
    )
    class_weights = compute_class_weights(train_loader, num_classes=num_classes)
    class_weights = class_weights.to(device)
    if use_focal_loss:
        loss_fn = FocalLoss(weight=class_weights)
    else:
        loss_fn = CrossEntropyLoss(weight=class_weights)
    trainer = Trainer(
        device=device,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_classes=num_classes,
    )

    trainer.train(
        train_loader,
        val_loader,
        use_early_stopping=early_stopping,
        epochs=epochs
    )
    trainer.plot_learning_curves()
    trainer.save_model(f"../models/saved/model_epoch_{epochs}_bs_{batch_size}_skip_{use_skip}_expansion_{expansion}_bn_{use_bn}.pth")
