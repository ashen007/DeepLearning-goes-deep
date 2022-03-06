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
    def rgb_channel_isolation(path, dst):
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

    @staticmethod
    def random_changes_to_color_properties(path, dst, delta=None, gamma_transformation=True, change_contrast=True,
                                           factor=None, steps=1):
        images = os.listdir(path)

        for image in images:
            image_path = os.path.join(path, image)
            img = img_to_array(PIL.Image.open(image_path))

            if delta is None:
                for i in range(steps):
                    change_factor = np.round(np.random.uniform(-1, 1), 2)
                    bc_img = array_to_img(tf.image.adjust_brightness(img, change_factor))
                    hue_img = array_to_img(tf.image.adjust_hue(img, change_factor))
                    sat_img = array_to_img(tf.image.adjust_saturation(img, change_factor))

                    # save transformed images
                    bc_img.save(os.path.join(dst, f'bc_{i}_{image}'))
                    hue_img.save(os.path.join(dst, f'hue_{i}_{image}'))
                    sat_img.save(os.path.join(dst, f'sat_{i}_{image}'))

            elif isinstance(delta, float):
                for i in range(steps):
                    change_factor = np.round(np.random.uniform(-1 * delta, 1 * delta), 2)
                    bc_img = array_to_img(tf.image.adjust_brightness(img, change_factor))
                    hue_img = array_to_img(tf.image.adjust_hue(img, change_factor))
                    sat_img = array_to_img(tf.image.adjust_saturation(img, change_factor))

                    # save transformed images
                    bc_img.save(os.path.join(dst, f'bc_{i}_{image}'))
                    hue_img.save(os.path.join(dst, f'hue_{i}_{image}'))
                    sat_img.save(os.path.join(dst, f'sat_{i}_{image}'))

            if factor is None:
                if gamma_transformation:
                    for i in range(steps):
                        gamma = np.round(np.random.uniform(1, 5), 2)
                        gamma_img = array_to_img(tf.image.adjust_gamma(img, gamma))
                        gamma_img.save(os.path.join(dst, f'gamma_img_{i}_{image}'))

                if change_contrast:
                    for i in range(steps):
                        change_factor = np.round(np.random.uniform(-5, 5), 2)
                        cont_img = array_to_img(tf.image.adjust_contrast(img, change_factor))
                        cont_img.save(os.path.join(dst, f'cont_img_{i}_{image}'))

            elif isinstance(factor, int):
                if gamma_transformation:
                    for i in range(steps):
                        gamma = np.round(np.random.uniform(1, factor), 2)
                        gamma_img = array_to_img(tf.image.adjust_gamma(img, gamma))
                        gamma_img.save(os.path.join(dst, f'gamma_img_{i}_{image}'))

                if change_contrast:
                    for i in range(steps):
                        factor = np.round(np.random.uniform(-1 * factor, factor), 2)
                        cont_img = array_to_img(tf.image.adjust_contrast(img, factor))
                        cont_img.save(os.path.join(dst, f'cont_img_{i}_{image}'))

            array_to_img(img).save(os.path.join(dst, image))

    @staticmethod
    def cropping(path, dst, central=True, random=False, fraction_low=0.5, fraction_high=0.9, random_width=224,
                 random_height=224, amount=3):
        images = os.listdir(path)

        for image in images:
            image_path = os.path.join(path, image)
            img = img_to_array(PIL.Image.open(image_path))

            if central:
                for i in range(amount):
                    crop_area = np.round(np.random.uniform(fraction_low, fraction_high), 2)
                    cc_img = tf.image.central_crop(img, central_fraction=crop_area)
                    array_to_img(cc_img).save(os.path.join(dst, f'cc_img_{i}_{image}'))

            elif random:
                for i in range(amount):
                    rc_img = tf.image.random_crop(img, size=[random_width, random_height, 3])
                    array_to_img(rc_img).save(os.path.join(dst, f'rc_img_{i}_{image}'))

            array_to_img(img).save(os.path.join(dst, image))

    @staticmethod
    def rotation(path, dst, angel=30, amount=3):
        images = os.listdir(path)

        for image in images:
            image_path = os.path.join(path, image)
            img = img_to_array(PIL.Image.open(image_path))
