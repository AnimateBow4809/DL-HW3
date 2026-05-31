import torch
import yaml
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from data.data_loader import MnistDataset
from models.inception_model import InceptionModel
from models.residual_model import ResidualModel
from models.resnext_model import ResNeXtModel
from utils.trainer import Trainer

from utils.config_utils import load_config, get_datasets, get_model
from utils.mnist_utils import load_mnist_from_pkl, load_fashion_mnist_raw

if __name__ == '__main__':
    config = load_config('../config/config.yaml')
    device = config['model']['device']
    arch = config['model']['arch']
    dataset = config['data']['dataset']
    batch_size = config['data']['batch_size']
    train_dataset, val_dataset, test_dataset = get_datasets(config)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = get_model(config)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(trainable_params, lr=config['optimizer']['lr'],
                                momentum=config['optimizer']['momentum'])
    trainer = Trainer(
        device=device,
        model=model,
        optimizer=optimizer,
        loss_fn=CrossEntropyLoss(),
    )

    trainer.train(train_loader, val_loader, use_early_stopping=config['model']['early_stopping'],
                  epochs=config['model']['epochs'])
    trainer.plot_learning_curves()
    trainer.plot_confusion_matrix(val_loader)
    trainer.save_model(f"../models/saved/{dataset}_model_{arch}.pth")
