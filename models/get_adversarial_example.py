import pickle
import os
import recoloradv.mister_ed.cifar10.cifar_loader as cifar_loader
import numpy as np
import yaml


def torch_data_matrix(loader):
    iterator = iter(loader)
    datamatrix = None
    labels = None
    while True:

        try:
            x, y = iterator.next()

            if datamatrix is None:
                datamatrix = np.moveaxis(x.numpy(), 1, -1)
                labels = y.numpy().reshape(-1, 1)
            else:
                x = np.moveaxis(x.numpy(), 1, -1)
                y = y.numpy().reshape(-1, 1)
                datamatrix = np.vstack((datamatrix, x))
                labels = np.vstack((labels, y))

        except StopIteration:
            break

    return datamatrix, labels


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def save_adversarial_examples(file, name):
    file_ = open(name+'.pickle', 'wb')
    pickle.dump(file, file_)
    file_.close()
    return


if __name__ == '__main__':
    cifar_valset = cifar_loader.load_cifar_data('val', batch_size=16)
    examples, labels = torch_data_matrix(cifar_valset)
    config = load_config(os.path.join(os.path.dirname(os.path.abspath(os.curdir)), 'config', 'config.yaml'))
    save_adversarial_examples(examples, os.path.join(config['generated']['path'], 'adversarial_examples'))
    save_adversarial_examples(labels, os.path.join(config['generated']['path'], 'adversarial_labels'))



