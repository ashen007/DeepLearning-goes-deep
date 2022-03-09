import tensorflow_addons as tfa
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
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
