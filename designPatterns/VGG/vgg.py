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


def learner(x, blocks):
    """ construct the feature extractor
    :param x: inputs to ther learner
    :param blocks: list of groups:: filter size and number of conv layers
    :return:
    """

    # the convolutional groups
    for n_layers, n_filters in blocks:
        x = group(x, n_layers, n_filters)

    return x


def group(x, n_layers, n_filters):
    """ construct a convolutional group
    :param x: input to the group
    :param n_layers: number of convolutional layers
    :param n_filters: number of filters
    :return:
    """

    # block of convolutional layers
    for n in range(n_layers):
        x = layers.Conv2D(n_filters, kernel_size=(3,3), strides=(1,1), padding='same', activation=activations.relu)(x)

    # max pooling at the end of the block
    x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    return x


def classifier(x, n_classes):
    """ construct the classifier
    :param x: input to the classifier
    :param n_classes: number of output classes
    :return:
    """
    # flatten the feature maps
    x = layers.Flatten()(x)

    # two fully connected dense layers
    x = layers.Dense(4096, activation=activations.relu)(x)
    x = layers.Dense(4096, activation=activations.relu)(x)

    # output layer for classification
    x = layers.Dense(n_classes, activation=activations.softmax)(x)

    return x
