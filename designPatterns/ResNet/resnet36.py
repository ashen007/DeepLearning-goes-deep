from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import losses


def steam(inputs):
    """ steam layers
    :param inputs: the input tensor
    :return:
    """
    outputs = layers.ZeroPadding2D(padding=(3, 3))(inputs)

    outputs = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.ReLU()(outputs)

    outputs = layers.ZeroPadding2D(padding=(1, 1))(outputs)
    outputs = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(outputs)

    return outputs
