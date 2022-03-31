import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D

from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam


def residual_block(x, filters, kernel=(3, 3), strid=(1, 1), projection=False):
    r = Conv2D(filters=filters, kernel_size=kernel, strides=strid, padding='same')(x)
    r = BatchNormalization()(r)
    r = Activation(activation=relu)(r)
    r = Conv2D(filters=filters, kernel_size=kernel, padding='same')(r)
    r = BatchNormalization()(r)

    if projection:
        skip = Conv2D(filters=filters, kernel_size=kernel, strides=strid, padding='same')(x)
        skip = BatchNormalization()(skip)
        r = Add()([r, skip])
    else:
        r = Add()([r, x])

    r = Activation(activation=relu)(r)

    return r


def build(input_shape, classes):
    input_layer = Input(input_shape)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)

    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)

    x = residual_block(x, filters=128, strid=(2, 2), projection=True)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)

    x = residual_block(x, filters=256, strid=(2, 2), projection=True)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=256)

    x = residual_block(x, filters=512, strid=(2, 2), projection=True)
    x = residual_block(x, filters=512)
    x = residual_block(x, filters=512)

    x = AveragePooling2D()(x)
    output_layer = Dense(units=classes, activation=softmax)(x)

    model = Model(input_layer, output_layer)
    model.compile(optimizer=Adam(learning_rate=0.1),
                  loss=categorical_crossentropy,
                  metrics=['accuracy', tfa.metrics.F1Score(num_classes=classes, threshold=0.5)])
    model.summary()

    return model
