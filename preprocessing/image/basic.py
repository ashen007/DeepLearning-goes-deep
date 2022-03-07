import os
import random

import numpy as np
import PIL.Image
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio

from tensorflow.keras.utils import array_to_img, img_to_array


def flip(path, dst, save_original=True):
    """
    :type save_original: bool
    :type dst: basestring
    :type path: basestring
    """
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

        if save_original:
            img.save(os.path.join(dst, image))


def rgb_channel_isolation(path, dst, save_original=True):
    """
    :type save_original: bool
    :type dst: basestring
    :type path: basestring
    """
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

        if save_original:
            # original
            img = array_to_img(img)
            img.save(os.path.join(dst, image))


def random_changes_to_color_properties(path, dst, delta=None, gamma_transformation=True, change_contrast=True,
                                       factor=None, copies=1, save_original=True):
    """
    :type copies: int
    :type factor: int
    :type change_contrast: bool
    :type gamma_transformation: bool
    :type delta: float
    :type save_original: bool
    :type dst: basestring
    :type path: basestring
    """
    images = os.listdir(path)

    for image in images:
        image_path = os.path.join(path, image)
        img = img_to_array(PIL.Image.open(image_path))

        if delta is None:
            for i in range(copies):
                change_factor = np.round(np.random.uniform(-1, 1), 2)
                bc_img = array_to_img(tf.image.adjust_brightness(img, change_factor))
                hue_img = array_to_img(tf.image.adjust_hue(img, change_factor))
                sat_img = array_to_img(tf.image.adjust_saturation(img, change_factor))

                # save transformed images
                bc_img.save(os.path.join(dst, f'bc_{i}_{image}'))
                hue_img.save(os.path.join(dst, f'hue_{i}_{image}'))
                sat_img.save(os.path.join(dst, f'sat_{i}_{image}'))

        elif isinstance(delta, float):
            for i in range(copies):
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
                for i in range(copies):
                    gamma = np.round(np.random.uniform(1, 5), 2)
                    gamma_img = array_to_img(tf.image.adjust_gamma(img, gamma))
                    gamma_img.save(os.path.join(dst, f'gamma_img_{i}_{image}'))

            if change_contrast:
                for i in range(copies):
                    change_factor = np.round(np.random.uniform(-5, 5), 2)
                    cont_img = array_to_img(tf.image.adjust_contrast(img, change_factor))
                    cont_img.save(os.path.join(dst, f'cont_img_{i}_{image}'))

        elif isinstance(factor, int):
            if gamma_transformation:
                for i in range(copies):
                    gamma = np.round(np.random.uniform(1, factor), 2)
                    gamma_img = array_to_img(tf.image.adjust_gamma(img, gamma))
                    gamma_img.save(os.path.join(dst, f'gamma_img_{i}_{image}'))

            if change_contrast:
                for i in range(copies):
                    factor = np.round(np.random.uniform(-1 * factor, factor), 2)
                    cont_img = array_to_img(tf.image.adjust_contrast(img, factor))
                    cont_img.save(os.path.join(dst, f'cont_img_{i}_{image}'))

        if save_original:
            array_to_img(img).save(os.path.join(dst, image))


def cropping(path, dst, central=True, random=False, fraction_low=0.5, fraction_high=0.9, random_width=224,
             random_height=224, copies=3, save_original=True):
    """
    :type copies: int
    :type random_height: int
    :type random_width: int
    :type fraction_high: float
    :type fraction_low: float
    :type random: bool
    :type central: bool
    :type save_original: bool
    :type dst: basestring
    :type path: basestring
    """
    images = os.listdir(path)

    for image in images:
        image_path = os.path.join(path, image)
        img = img_to_array(PIL.Image.open(image_path))

        if central:
            for i in range(copies):
                crop_area = np.round(np.random.uniform(fraction_low, fraction_high), 2)
                cc_img = tf.image.central_crop(img, central_fraction=crop_area)
                array_to_img(cc_img).save(os.path.join(dst, f'cc_img_{i}_{image}'))

        elif random:
            for i in range(copies):
                rc_img = tf.image.random_crop(img, size=[random_width, random_height, 3])
                array_to_img(rc_img).save(os.path.join(dst, f'rc_img_{i}_{image}'))

        if save_original:
            array_to_img(img).save(os.path.join(dst, image))


def rotation(path, dst, angel=30, copies=3, save_original=True):
    """
    :type angel: int
    :type copies: int
    :type save_original: bool
    :type dst: basestring
    :type path: basestring
    """
    images = os.listdir(path)

    for image in images:
        image_path = os.path.join(path, image)
        img = img_to_array(PIL.Image.open(image_path))

        for i in range(copies):
            random_angel = np.random.randint(-1 * angel, angel)
            ra_img = tfa.image.rotate(img, random_angel)
            array_to_img(ra_img).save(os.path.join(dst, f'ra_img_{i}_{image}'))

        if save_original:
            array_to_img(img).save(os.path.join(dst, image))


def translation(path, dst, height_shift_range=0.3, width_shift_range=0.3, height_shift=True, width_shift=True,
                copies=3, save_original=True):
    """
    :type copies: int
    :type width_shift: bool
    :type height_shift: bool
    :type width_shift_range: float
    :type height_shift_range: float
    :type save_original: bool
    :type dst: basestring
    :type path: basestring
    """
    images = os.listdir(path)

    for image in images:
        image_path = os.path.join(path, image)
        img = img_to_array(PIL.Image.open(image_path))

        for i in range(copies):
            if height_shift and width_shift:
                random_hshift = np.random.uniform(0.1, height_shift_range)
                random_wshift = np.random.uniform(0.1, width_shift_range)
                rs_img = (tf.keras.preprocessing.image.random_shift(img, random_wshift, random_hshift, channel_axis=2,
                                                                    row_axis=0, col_axis=1,
                                                                    fill_mode='reflect')).astype(np.uint8)
                array_to_img(rs_img).save(os.path.join(dst, f'rs_img_{i}_{image}'))

            elif height_shift and not width_shift:
                random_hshift = np.random.uniform(0.1, height_shift_range)
                rs_img = (tf.keras.preprocessing.image.random_shift(img, 0, random_hshift, channel_axis=2,
                                                                    row_axis=0, col_axis=1,
                                                                    fill_mode='reflect')).astype(np.uint8)
                array_to_img(rs_img).save(os.path.join(dst, f'rs_h_img_{i}_{image}'))

            elif not height_shift and width_shift:
                random_wshift = np.random.uniform(0.1, width_shift_range)
                rs_img = (tf.keras.preprocessing.image.random_shift(img, random_wshift, 0, channel_axis=2,
                                                                    row_axis=0, col_axis=1,
                                                                    fill_mode='reflect')).astype(np.uint8)
                array_to_img(rs_img).save(os.path.join(dst, f'rs_img_{i}_{image}'))

        if save_original:
            array_to_img(img).save(os.path.join(dst, image))


def noise_injection(path, dst, magnitude=0.5, copies=3, save_original=True):
    """
    :type copies: int
    :type magnitude: float
    :type save_original: bool
    :type dst: basestring
    :type path: basestring
    """
    images = os.listdir(path)

    for image in images:
        image_path = os.path.join(path, image)
        img = img_to_array(PIL.Image.open(image_path)) / 255.

        for i in range(copies):
            random_magnitude = np.round(np.random.uniform(0, magnitude), 2)
            noise = np.random.normal(0, np.round(random_magnitude, decimals=3), size=img.shape)
            ni_img = array_to_img((img + noise))
            array_to_img(ni_img).save(os.path.join(dst, f'ni_img_{i}_{image}'))

        if save_original:
            array_to_img(img).save(os.path.join(dst, image))
