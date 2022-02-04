import tensorflow as tf


class Architecture:
    def __init__(self, classes, freeze):
        self.classes = classes
        self.freeze = freeze

    def returned_pretrained_resnet50(self):
        base_model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet')
        avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(self.classes, activation='softmax')(avg)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)

        # Setting the lower layers trainable as False
        if self.freeze:
            for layer in base_model.layers:
                layer.trainable = False
            return model

        return model