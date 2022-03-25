import tensorflow_addons as tfa
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Input, concatenate
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy


class Inception(Layer):

    def __init__(self, f_1x1, f_3x3, f_3x3_reduce, f_5x5, f_5x5_reduce, projection, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.f_1x1 = f_1x1
        self.f_3x3 = f_3x3
        self.f_3x3_reduce = f_3x3_reduce
        self.f_5x5 = f_5x5
        self.f_5x5_reduce = f_5x5_reduce
        self.projection = projection

    def build(self, input_shape):
        self.pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.x1 = Conv2D(filters=self.f_1x1, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding='same')
        self.x3_reducer = Conv2D(filters=self.f_3x3_reduce, kernel_size=(1, 1), strides=(1, 1), activation=relu,
                                 padding='same')
        self.x5_reducer = Conv2D(filters=self.f_5x5_reduce, kernel_size=(1, 1), strides=(1, 1), activation=relu,
                                 padding='same')
        self.x3 = Conv2D(filters=self.f_3x3, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding='same')
        self.x5 = Conv2D(filters=self.f_5x5, kernel_size=(5, 5), strides=(1, 1), activation=relu, padding='same')
        self.proj = Conv2D(filters=self.projection, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding='same')

    def call(self, inputs, *args, **kwargs):
        pool_layer = self.pool(inputs)
        skip_path = self.x1(inputs)
        conv_3_red = self.x3_reducer(inputs)
        conv_5_red = self.x5_reducer(inputs)
        conv_3 = self.x3(conv_3_red)
        conv_5 = self.x5(conv_5_red)
        conv_pool = self.proj(pool_layer)
        depth_concat = concatenate([conv_pool, conv_3, conv_5, skip_path], axis=3)

        return depth_concat


def model_builder(shape, classes):
    model = Sequential([Input(shape=shape),
                        Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), activation=relu, padding='same'),
                        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
                        BatchNormalization(),
                        Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding='same'),
                        Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding='same'),
                        BatchNormalization(),
                        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
                        Inception(64, 128, 96, 32, 16, 32, name='inception_3a'),
                        Inception(128, 192, 128, 96, 32, 64, name='inception_3b'),
                        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
                        Inception(192, 208, 96, 48, 16, 64, name='inception_4a'),

                        # aux_1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
                        # aux_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding='valid')(aux_1)
                        # aux_1 = Dense(units=1024, activation=relu)(aux_1)
                        # aux_1 = Dropout(rate=0.7)(aux_1)
                        # aux_1 = Flatten()(aux_1)
                        # aux_out1 = Dense(units=classes, activation=softmax, name='aux_out1')(aux_1)

                        Inception(160, 224, 112, 64, 24, 64, name='inception_4b'),
                        Inception(128, 256, 128, 64, 24, 64, name='inception_4c'),
                        Inception(112, 288, 144, 64, 32, 64, name='inception_4d'),
                        Inception(256, 320, 160, 128, 32, 128, name='inception_4e'),

                        # aux_2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
                        # aux_2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding='valid')(aux_2)
                        # aux_2 = Dense(units=1024, activation=relu)(aux_2)
                        # aux_2 = Dropout(rate=0.7)(aux_2)
                        # aux_2 = Flatten()(aux_2)
                        # aux_out2 = Dense(units=classes, activation=softmax, name='aux_out2')(aux_2)

                        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
                        Inception(256, 320, 160, 128, 32, 128, name='inception_5a'),
                        Inception(384, 384, 192, 128, 48, 128, name='inception_5b'),
                        AveragePooling2D(pool_size=(7, 7), strides=(1, 1)),
                        Dropout(rate=0.4),
                        Flatten(),
                        Dense(units=classes, activation=softmax, name='main_out')])

    model.compile(optimizer=Adam(), loss=categorical_crossentropy,
                  loss_weights={'main_out': 1, 'aux_out1': 0.3, 'aux_out2': 0.3},
                  metrics=['accuracy', tfa.metrics.F1Score(num_classes=classes, threshold=0.5)])
    model.summary()

    return model
