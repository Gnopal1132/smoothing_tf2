import tensorflow as tf
import numpy as np
import pickle
import os
from tqdm import tqdm


class GaussianDataset:
    def __init__(self, config):
        self.config = config
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = self.give_dataset()

    @staticmethod
    def split_dataset(X, Y):
        trainsize = int(len(X) * 0.8)
        x_train, y_train, x_val, y_val = X[:trainsize], Y[:trainsize], X[trainsize:], Y[trainsize:]
        return x_train, y_train, x_val, y_val

    @staticmethod
    def give_dataset():
        cifar = tf.keras.datasets.cifar10
        (X_train, Y_train), (X_test, Y_test) = cifar.load_data()
        return (X_train, Y_train), (X_test, Y_test)

    def save_examples(self, file, name):
        file_ = open(name + '.pickle', 'wb')
        pickle.dump(file, file_)
        file_.close()
        return

    def gaussian_noise_data(self):
        final_x = None
        final_y = []
        sigma = self.config['smoothing']['sigma']

        for idx, x in tqdm(enumerate(self.X_train)):
            num = self.config['smoothing']['N']

            batch = tf.tile(tf.expand_dims(x, axis=0), multiples=[num, 1, 1, 1])
            noise = tf.random.normal(batch.shape, dtype=tf.float16) * sigma
            instance = tf.cast(batch, dtype=tf.float16) + noise

            if final_x is None:
                final_x = instance.numpy()
            else:
                final_x = np.vstack((final_x, instance))

            final_y.extend([self.Y_train[idx]] * num)

            if (idx % 10000 == 0 and idx != 0) or idx+1 == len(self.X_train):
                self.save_examples(final_x,
                                   name=os.path.join(self.config['generated']['path'],
                                                     f'adversarial_example_{idx}'))
                self.save_examples(final_y, name=os.path.join(self.config['generated']['path'],
                                                     f'adversarial_label_{idx}'))
                final_x = None
                final_y.clear()



