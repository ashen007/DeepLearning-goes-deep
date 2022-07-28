from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Normalization


def pre_stream(inputs):
    """ pre-steam layers
    :param inputs: input tensor
    :return:
    """
    outputs = Normalization()(inputs)

    return outputs


def model(inputs):
    """ model constructor
    :param inputs:
    :return:
    """
    pass


# create empty wrapper model
exist_model = model(inputs=None)
wrapper = Sequential()

# add pre-stream first to wrapper
wrapper.add(wrapper)

# add existing model to wrapper
wrapper.add(exist_model)
