import tensorflow as tf
import os
import glob
import yaml
import pickle
import gc


class DataLoader:
    PREVIOUS_DATA = None
    PREVIOUS_PATH = None
    PREVIOUS_LABEL = None
    PREVIOUS_PATH_LABEL = None

    def __init__(self, config):
        self.config = config
        self.path = self.config['generated']['colabpath']
        self.batch = self.config['train']['batch_size']
        self.buffer = self.config['train']['buffer']
        self.full_path_data, self.full_path_label = self.return_paths()
        self.temp_train, self.temp_val = self.split_dataset(self.full_path_data, self.full_path_label)
        self.train = tf.data.Dataset.from_tensor_slices(self.temp_train)
        self.val = tf.data.Dataset.from_tensor_slices(self.temp_val)

    @staticmethod
    def split_dataset(X, Y):
        trainsize = int(len(X) * 0.8)
        x_train, y_train, x_val, y_val = X[:trainsize], Y[:trainsize], X[trainsize:], Y[trainsize:]
        final_train = []
        final_val = []
        for train, label in zip(x_train, y_train):
            final_train.append([train, label])

        for val, label in zip(x_val, y_val):
            final_val.append([val, label])

        return final_train, final_val

    def return_paths(self):
        file_length = [200020, 200000, 200000, 200000, 199980]  #Manually calculated
        idx = 0
        full_path = []
        full_label_path = []
        files = glob.glob(os.path.join(self.path, 'adversarial_example_*.pickle'), recursive=True)
        labels = glob.glob(os.path.join(self.path, 'adversarial_label_*.pickle'), recursive=True)
        for instance, label in zip(files, labels):
            full_path.extend([os.path.join(instance, str(i)) for i in range(file_length[idx])])
            full_label_path.extend([os.path.join(label, str(i)) for i in range(file_length[idx])])
            idx += 1

        return full_path, full_label_path

    @staticmethod
    def return_element_index_data(path, index):

        if DataLoader.PREVIOUS_PATH is None:
            file = open(path, 'rb')
            result = pickle.load(file)
            DataLoader.PREVIOUS_DATA = result
            DataLoader.PREVIOUS_PATH = path
            file.close()
            return result[index]

        elif DataLoader.PREVIOUS_PATH == path:
            return DataLoader.PREVIOUS_DATA[index]
        else:
            DataLoader.PREVIOUS_DATA = None
            DataLoader.PREVIOUS_PATH = None
            gc.collect()
            file = open(path, 'rb')
            result = pickle.load(file)
            DataLoader.PREVIOUS_DATA = result
            DataLoader.PREVIOUS_PATH = path
            file.close()
            return result[index]

    @staticmethod
    def return_element_index_label(path, index):

        if DataLoader.PREVIOUS_PATH_LABEL is None:
            file = open(path, 'rb')
            result = pickle.load(file)
            DataLoader.PREVIOUS_LABEL = result
            DataLoader.PREVIOUS_PATH_LABEL = path
            file.close()
            return result[index]

        elif DataLoader.PREVIOUS_PATH_LABEL == path:
            return DataLoader.PREVIOUS_LABEL[index]
        else:
            DataLoader.PREVIOUS_LABEL = None
            DataLoader.PREVIOUS_PATH_LABEL = None
            gc.collect()
            file = open(path, 'rb')
            result = pickle.load(file)
            DataLoader.PREVIOUS_LABEL = result
            DataLoader.PREVIOUS_PATH_LABEL = path
            file.close()
            return result[index]

    def load_img_label(self, instance):
        data_path = instance[0].numpy()
        label_path = instance[1].numpy()

        file_to_open = os.path.dirname(data_path)
        label_to_open = os.path.dirname(label_path)
        index_data = int(os.path.basename(data_path))

        data, label = self.return_element_index_data(file_to_open, index_data), \
                      self.return_element_index_label(label_to_open, index_data)

        return tf.cast(data, dtype=tf.float16), tf.cast(label, dtype=tf.float16)

    def preprocess(self, instance):
        img, label = tf.py_function(self.load_img_label, [instance], [tf.float16, tf.float16])
        return tf.ensure_shape(img, [None, None, 3]), tf.ensure_shape(label, [None, ])

    def final_loader(self, dataset, BATCH_SIZE=2, BUFFER_SIZE=2):
        data = dataset.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        data = data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat(1)
        data = data.prefetch(buffer_size=tf.data.AUTOTUNE)
        return data

    def return_loader(self):
        train = self.final_loader(self.train, BATCH_SIZE=self.batch, BUFFER_SIZE=self.buffer)
        val = self.final_loader(self.val, BATCH_SIZE=self.batch, BUFFER_SIZE=self.buffer)
        return train, val


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


if __name__ == '__main__':
    config = load_config(os.path.join(os.path.dirname(os.path.abspath(os.curdir)), 'config', 'config.yaml'))
    train_loader, val_loader = DataLoader(config).return_loader()
    for X, Y in train_loader.take(1):
        print(X.shape)
        print(Y.shape)

