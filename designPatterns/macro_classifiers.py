from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import losses


def single_classifier(x, n_classes):
    """ construct the single output classifier
    :param x: input to the classifier
    :param n_classes: number of output classes
    :return:
    """
    # pool at the end of all the convolutional layers
    x = layers.GlobalAveragePooling2D()(x)

    # final dense layer for the outputs
    outputs = layers.Dense(n_classes, activation=activations.softmax, kernel_initializer='he_normal')(x)

    return outputs


def multi_classifier(x, n_classes):
    """ construct the multi output classifier
    :param x: input to the classifier
    :param n_classes: number of output classes
    :return:
    """
    # high dimensionality feature extraction (encoding)
    encoding = x

    # low dimensionality feature extraction (embedding)
    embedding = layers.GlobalAveragePooling2D()(x)

    # pre activation probabilities (soft labels)
    probabilities = layers.Dense(n_classes)(embedding)

    # post activation probabilities (hard labels)
    outputs = layers.Activation(activation=activations.softmax)(probabilities)

    return encoding, embedding, probabilities, outputs
