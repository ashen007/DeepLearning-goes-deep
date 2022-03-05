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
    def flip(path, dst):
        images = os.listdir(path)

        for image in images:
            image_path = os.path.join(path, image)
            dst_lr_path = os.path.join(dst, f'_flip_l_r_{image}')
            dst_ud_path = os.path.join(dst, f'_flip_u_d_{image}')
            img = PIL.Image.open(image_path)
            flip_lr = array_to_img(tf.image.flip_left_right(img_to_array(img)))
            flip_ud = array_to_img(tf.image.flip_up_down(img_to_array(img)))

            # save flipped image in destination
            flip_lr.save(dst_lr_path)
            flip_ud.save(dst_ud_path)
            img.save(os.path.join(dst, image))
