import tensorflow_addons as tfa
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam


def inception(x, filters, projection, classes=None, aux=False, name=None, aux_name=None):
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
    x = inception(x, [64, 128, 96, 32, 16], projection=32, name='inception_3a')
    x = inception(x, [128, 192, 128, 96, 32], projection=64, name='inception_3b')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = inception(x, [192, 208, 96, 48, 16], projection=64, name='inception_4a')

    aux_1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
    aux_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding='valid')(aux_1)
    aux_1 = Dense(units=1024, activation=relu)(aux_1)
    aux_1 = Dropout(rate=0.7)(aux_1)
    aux_1 = Flatten()(aux_1)
    aux_out1 = Dense(units=classes, activation=softmax, name='aux_out1')(aux_1)

    x = inception(x, [160, 224, 112, 64, 24], projection=64, name='inception_4b')
    x = inception(x, [128, 256, 128, 64, 24], projection=64, name='inception_4c')
    x = inception(x, [112, 288, 144, 64, 32], projection=64, name='inception_4d')
    x = inception(x, [256, 320, 160, 128, 32], projection=128, name='inception_4e')

    aux_2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
    aux_2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding='valid')(aux_2)
    aux_2 = Dense(units=1024, activation=relu)(aux_2)
    aux_2 = Dropout(rate=0.7)(aux_2)
    aux_2 = Flatten()(aux_2)
    aux_out2 = Dense(units=classes, activation=softmax, name='aux_out2')(aux_2)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = inception(x, [256, 320, 160, 128, 32], projection=128, name='inception_5a')
    x = inception(x, [384, 384, 192, 128, 48], projection=128, name='inception_5b')
    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)
    x = Dropout(rate=0.4)(x)
    x = Flatten()(x)
    output_layer = Dense(units=classes, activation=softmax, name='main_out')(x)

    model = Model(input_layer, [output_layer, aux_out1, aux_out2])
    model.compile(optimizer=Adam(), loss=categorical_crossentropy,
                  loss_weights={'main_out': 1, 'aux_out1': 0.3, 'aux_out2': 0.3},
                  metrics=['accuracy', tfa.metrics.F1Score(num_classes=classes, threshold=0.5)])
    model.summary()

    return model
