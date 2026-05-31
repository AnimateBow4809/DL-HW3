import torch
import yaml
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.inception_model import InceptionModel
from models.residual_model import ResidualModel
from utils.config_utils import get_datasets, get_model
from utils.trainer import Trainer


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    config = load_config('../config/config.yaml')
    train_dataset, val_dataset, test_dataset = get_datasets(config)
    y_train = train_dataset.y
    batch_size = config['data']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = config['model']['device']
    arch = config['model']['arch']
    model = get_model(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['optimizer']['lr'],
                                momentum=config['optimizer']['momentum'])
    trainer = Trainer(
        device=device,
        model=model,
        optimizer=optimizer,
        loss_fn=CrossEntropyLoss(),
    )
    trainer.plot_feature_maps(test_loader)
    trainer.plot_confusion_matrix(test_loader)
    trainer.plot_misclassified_predictions(test_loader)
    test_loss, test_acc = trainer.evaluate(test_loader)
    per_class_accuracy = trainer.get_per_class_accuracy(test_loader)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    for i in range(10):
        print(f"Class {i} Accuracy: {per_class_accuracy[i]['accuracy']}, "
              f"Total Samples: {torch.sum(torch.from_numpy(y_train)==i)}")
