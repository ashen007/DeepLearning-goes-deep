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


def learner(x, groups):
    """ construct the learner
    :param x: input to the learner
    :param groups: list of groups :: number of filters and blocks
    :return:
    """
    # first residual block group (not stride)
    n_filters, n_blocks = groups.pop(0)
    x = group(x, n_filters, n_blocks, strides=(1, 1))

    # remaining residual block groups (stride)
    for n_filters, n_blocks in groups:
        x = group(x, n_filters, n_blocks)

    return x


def group(x, n_filters, n_blocks, strides=(2, 2)):
    """ construct a residual group
    :param x: input into the group
    :param n_filters: number of filters for the group
    :param n_blocks: number of residual blocks with identity link
    :param strides: weather the projection block is a stride convolution
    :return:
    """
    # double the size of filters to fit the first residual group
    x = projection_block(x, n_filters, strides=strides)

    # identity residual blocks
    for _ in range(n_blocks):
        x = identity_block(x, n_filters)

    return x


def identity_block(x, n_filters):
    """ construct a bottleneck residual block with identity link
    :param x: input into the block
    :param n_filters: number of filters
    :return:
    """
    # save input vector for identity link
    shortcut = x

    # construct the 1x1, 3x3, 1x1 residual block

    # dimensionality reduction
    x = layers.Conv2D(n_filters, kernel_size=(1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # bottleneck layer
    x = layers.Conv2D(n_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # dimensionality restoration - increase the number of output filters by 4x
    x = layers.Conv2D(n_filters * 4, kernel_size=(1, 1), strides=(1, 1), use_bias=False,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    # add the identity link (input) to the output of the residual block
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)

    return x


def projection_block(x, n_filters, strides=(2, 2)):
    """ construct bottleneck residual block with projection shortcut
        increase the number of filters by 4x
    :param x: input into the block
    :param n_filters: number of filters
    :param strides: whether entry convolution is stride
    :return:
    """
    # construct the projection shortcut
    # increase filters by 4x to match shape when added to the output block
    shortcut = layers.Conv2D(4 * n_filters, kernel_size=(1, 1), strides=strides, use_bias=False,
                             kernel_initializer='he_normal')(x)
    shortcut = layers.BatchNormalization()(shortcut)

    # construct the 1x1, 3x3, 1x1 residual block

    # dimensionality reduction
    # feature pooling when strides=(2,2)
    x = layers.Conv2D(n_filters, kernel_size=(1, 1), strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # bottleneck layer
    x = layers.Conv2D(n_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # dimensionality restoration - increase the number of filters by 4x
    x = layers.Conv2D(4 * n_filters, kernel_size=(1, 1), strides=(1, 1), use_bias=False,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    # add the projection shortcut link to the output of the residual block
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x
