from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import losses


def steam(inputs, levels=1):
    """ steam layers
    :param levels: how many layers need to initiate in steam
    :param inputs: the input tensor
    :return:
    """
    outputs = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                            activation=activations.relu)(inputs)

    if levels > 1:
        for i in range(levels):
            outputs = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                    activation=activations.relu)(outputs)

    return outputs
