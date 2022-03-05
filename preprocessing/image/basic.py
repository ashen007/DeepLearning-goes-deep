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

    @staticmethod
    def rgb_channel_isolation(self, path, dst):
        images = os.listdir(path)

        for image in images:
            image_path = os.path.join(path, image)
            img = img_to_array(PIL.Image.open(image_path))
            r, g, b = img, img, img
            r[:, :, 1:3] = 0
            g[:, :, 0] = 0
            g[:, :, 2] = 0
            b[:, :, 0:2] = 0

            # red component
            r = array_to_img(r)
            r.save(os.path.join(dst, f'_isolate_r_{image}'))
            # green component
            g = array_to_img(g)
            g.save(os.path.join(dst, f'_isolate_g_{image}'))
            # blue component
            b = array_to_img(b)
            b.save(os.path.join(dst, f'_isolate_b_{image}'))
            # original
            img = array_to_img(img)
            img.save(os.path.join(dst, image))
