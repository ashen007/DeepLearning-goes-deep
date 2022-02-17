from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.regularizers import l2

optimizer = SGD(learning_rate=1e-2, momentum=9e-1)
weight_decay = 5e-4


def vgg_nett16(input_shape=(224, 224, 3), classes=None):
    # input layer
    input_layer = Input(shape=input_shape, name='input_')

    # first conv block
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(weight_decay), padding='same',
               activation=relu)(input_layer)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(weight_decay), padding='same',
               activation=relu)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # second conv block
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(weight_decay), padding='same',
               activation=relu)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(weight_decay), padding='same',
               activation=relu)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # third conv block
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(weight_decay), padding='same',
               activation=relu)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(weight_decay), padding='same',
               activation=relu)(x)
    x = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), kernel_regularizer=l2(weight_decay), padding='same',
               activation=relu)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # fourth conv block
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(weight_decay), padding='same',
               activation=relu)(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(weight_decay), padding='same',
               activation=relu)(x)
    x = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), kernel_regularizer=l2(weight_decay), padding='same',
               activation=relu)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # fifth conv block
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(weight_decay), padding='same',
               activation=relu)(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(weight_decay), padding='same',
               activation=relu)(x)
    x = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), kernel_regularizer=l2(weight_decay), padding='same',
               activation=relu)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # classifier
    x = Flatten()(x)
    x = Dense(units=512, activation=relu)(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=512, activation=relu)(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=classes, activation=softmax)(x)

    model = Model(input_layer, x)
    model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()

    return model
