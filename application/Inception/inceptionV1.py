import tensorflow_addons as tfa
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dropout

from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam


def inception(x, filters, projection, name=None):
    f_1x1, f_3x3, f_3x3_reduce, f_5x5, f_5x5_reduce = filters
    x1 = Conv2D(filters=f_1x1, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding='same')(x)
    x3_reducer = Conv2D(filters=f_3x3_reduce, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding='same')(x)
    x5_reducer = Conv2D(filters=f_5x5_reduce, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding='same')(x)
    pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)

    x3 = Conv2D(filters=f_3x3, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding='same')(x3_reducer)
    x5 = Conv2D(filters=f_5x5, kernel_size=(5, 5), strides=(1, 1), activation=relu, padding='same')(x5_reducer)
    proj = Conv2D(filters=projection, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding='same')(pool)

    x = concatenate([x1, x3, x5, proj], axis=3, name=name)

    return x


def model_builder(shape, classes):
    input_layer = Input(shape=shape)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), activation=relu, padding='same')(input_layer)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding='same')(x)
    x = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = inception(x, [64, 128, 96, 32, 16], projection=32, name='inception 3a')(x)
    x = inception(x, [128, 192, 128, 96, 32], projection=64, name='inception 3b')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = inception(x, [192, 208, 96, 48, 16], projection=64, name='inception 4a')
    x = inception(x, [160, 224, 112, 64, 24], projection=64, name='inception 4b')
    x = inception(x, [128, 256, 128, 64, 24], projection=64, name='inception 4c')
    x = inception(x, [112, 288, 144, 64, 32], projection=64, name='inception 4d')
    x = inception(x, [256, 320, 160, 128, 32], projection=128, name='inception 4e')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = inception(x, [256, 320, 160, 128, 32], projection=128, name='inception 5a')
    x = inception(x, [384, 384, 192, 128, 48], projection=128, name='inception 5b')
    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)
    x = Dropout(rate=0.4)(x)
    output_layer = Dense(units=classes, activation=softmax)(x)

    model = Model(input_layer, output_layer)
    model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy', tfa.metrics.F1Score])
    model.summary()

    return model
