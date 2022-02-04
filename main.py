import os
from models.dataset_gaussian_noise import GaussianDataset
from models.train import Trainer
from models.dataloader import DataLoader
import pickle
import yaml


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def save_examples(file, name):
    file_ = open(name + '.pickle', 'wb')
    pickle.dump(file, file_)
    file_.close()
    return


if __name__ == '__main__':
    # Lets load the data
    # Initiating the Trainer Module
    config = load_config(os.path.join(os.path.abspath(os.curdir), 'config', 'config.yaml'))
    print(os.path.join(os.path.abspath(os.curdir), 'config', 'config.yaml'))

    dataset = GaussianDataset(config)
    dataset.gaussian_noise_data()

    """train_loader, val_loader = DataLoader(config).return_loader()

    trainer = Trainer(config=config, train=train_loader, val=val_loader,
                      classes=config['dataset']['classes'])
    # Training the Module with freezing
    trainer.train_model(freeze=True, epochs=30)

    # Training With all the weights unfreezed
    trainer.train_model(freeze=False, epochs=100)"""

    # Lets predict using randomized smoothing

    """# Lets Load the Adversarial Examples
    X_adv = adv_examples(os.path.join(config['generated']['path'], 'adversarial_examples.pickle'))
    Y_adv = adv_examples(os.path.join(config['generated']['path'], 'adversarial_labels.pickle'))
    run_prediction(X_adv, Y_adv)"""

