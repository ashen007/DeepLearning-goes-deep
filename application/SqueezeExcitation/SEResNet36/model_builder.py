import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalAveragePooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import ReLU

from tensorflow.keras.metrics import Recall
from tensorflow.keras.activations import relu, softmax, sigmoid
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


def squeeze_excitation_block(x, out_dim, ratio):
    s = GlobalAveragePooling1D()(x)
    e = Dense(units=out_dim / ratio)(s)
    e = ReLU()(e)
    e = Dense(units=out_dim)(e)
    e = Activation(activation=sigmoid)(e)
    e = tf.reshape(e, [-1, 1, 1, out_dim])

    scale = x * e

    return scale


def build(input_shape, classes):
    input_layer = Input(input_shape)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')(input_layer)
    x = BatchNormalization()(x)
    x = Activation(activation=relu)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = squeeze_excitation_block(x, out_dim=256, ratio=16)

    x = residual_block(x, filters=128, strid=(2, 2), projection=True)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = squeeze_excitation_block(x, out_dim=512, ratio=16)

    x = residual_block(x, filters=256, strid=(2, 2), projection=True)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=256)
    x = squeeze_excitation_block(x, out_dim=1024, ratio=16)

    x = residual_block(x, filters=512, strid=(2, 2), projection=True)
    x = residual_block(x, filters=512)
    x = residual_block(x, filters=512)
    x = squeeze_excitation_block(x, out_dim=2048, ratio=16)

    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    output_layer = Dense(units=classes, activation=softmax)(x)

    model = Model(input_layer, output_layer)
    model.compile(optimizer=Adam(),
                  loss=categorical_crossentropy,
                  metrics=['accuracy', Recall()])
    model.summary()

    return model
