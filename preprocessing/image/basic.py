import os
import numpy as np
import PIL.Image
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio

from tensorflow.keras.utils import array_to_img, img_to_array


class BasicAugmentation:
    def __init__(self):
        return

    @staticmethod
    def random_horizontal_flip(path, dst):
        images = os.listdir(path)

        for image in images:
            image_path = os.path.join(path, image)
            dst_path = os.path.join(dst, image)
            img = PIL.Image.open(image_path)
            flip = tf.image.flip_left_right(img)

            # save flipped image in destination
            flip.save(dst_path)
