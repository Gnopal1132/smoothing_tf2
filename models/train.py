# Train the Base Classifier
import os.path
import tensorflow as tf
from models.architecture import Architecture


class Trainer:
    def __init__(self, config,
                 train,
                 val,
                 classes):
        """

        :param config:
        :param train_data:
        :param train_label:
        :param val_data:
        :param val_label:
        :param classes:
        """
        self.config = config
        self.train = train
        self.val = val
        self.classes = classes
        self.model = Architecture(self.classes, freeze=True).returned_pretrained_resnet50()
        self.callbacks = self.return_callbacks(self.config)

    @staticmethod
    def get_id(path):
        import time
        id = time.strftime('run_%Y_%m_%d_%H_%M_%S')
        return os.path.join(path, id)

    def return_callbacks(self, config):
        callbacks = []
        # Early Stopping
        if config['train']['callbacks']['earlystop']['use']:
            patience = config['train']['callbacks']['earlystop']['patience']
            earlystop = tf.keras.callbacks.EarlyStopping(patience=patience,
                                                         restore_best_weights=True)
            callbacks.append(earlystop)

        if config['train']['callbacks']['tensorboard']['use']:
            path = self.get_id(config['train']['callbacks']['tensorboard']['path'])
            board = tf.keras.callbacks.TensorBoard(log_dir=path)
            callbacks.append(board)

        if config['train']['callbacks']['checkpoint']['last_checkpoint']['use']:
            path = config['train']['callbacks']['checkpoint']['last_checkpoint']['path']
            last_cp = tf.keras.callbacks.ModelCheckpoint(path, save_best_only=False, save_weight_only=True)
            callbacks.append(last_cp)

        if config['train']['callbacks']['checkpoint']['best_checkpoint']['use']:
            path = config['train']['callbacks']['checkpoint']['best_checkpoint']['path']
            best_cp = tf.keras.callbacks.ModelCheckpoint(path, save_best_only=True, save_weight_only=True)
            callbacks.append(best_cp)

        if config['train']['callbacks']['scheduler']['reducelr']['use']:
            factor = config['train']['callbacks']['scheduler']['reducelr']['factor']
            patience = config['train']['callbacks']['scheduler']['reducelr']['factor']
            earlystop = tf.keras.callbacks.ReduceLROnPlateau(factor=factor, monitor='val_loss',
                                                             patience=patience)
            callbacks.append(earlystop)

        return callbacks

    def save_graph(self, model, graph_path):
        graph = model.to_json()
        with open(graph_path, 'w') as file:
            file.write(graph)

    def train_model(self, epochs, freeze=True):
        if freeze:
            self.fit(epochs)
        else:
            for layers in self.model.layers:
                layers.trainable = True
            self.fit(epochs)

    def fit(self, epochs):

        # Use Pretrained Weights
        if self.config["train"]["weight_initialization"]["use_pretrained"]:
            read_from = self.config["train"]["weight_initialization"]["restore_from"]
            print("Restoring Weights From: ", read_from)
            self.model.load_weights(read_from)
        else:
            graph_path = self.config['generated']['graph_path']
            print("Saving Weights", graph_path)
            self.save_graph(self.model, graph_path)

        # Compiling the model
        opt = self.config["train"]["optimizer"]
        lr = self.config["train"]["learning_rate"]

        if opt == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                                 amsgrad=False)
        elif opt == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
        elif opt == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
        elif opt == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(lr=lr, epsilon=None, decay=0.0)
        elif opt == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)
        else:
            raise Exception('Optimizer unknown')

        self.model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=optimizer,
                           metrics=['accuracy'])

        use_multi_processing = self.config["train"]["use_multiprocessing"]
        self.model.fit(self.train, epochs=epochs,
                       validation_data=self.val,
                       callbacks=self.return_callbacks(self.config),
                       batch_size=self.config['train']['batch_size'],
                       use_multiprocessing=use_multi_processing)
        # save weights
        out = self.config['train']['output']['weight']
        print("Saving weights in", out)
        self.model.save(out)
