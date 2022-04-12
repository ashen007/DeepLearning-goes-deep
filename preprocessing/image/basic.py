import os
import random

import numpy as np
import PIL.Image
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio

from tensorflow.keras.utils import array_to_img, img_to_array


def load_image(path, mode='RGB'):
    return PIL.Image.open(path, mode=mode)


def to_array(image):
    return np.asarray(image)


def to_image(array, mode='RGB'):
    return PIL.Image.fromarray(np.uint8(array), mode=mode)


def change_rgb_to(image, mode='grayscale'):
    if mode == 'grayscale':
        temp = tf.image.rgb_to_grayscale(image)
        return temp

    elif mode == 'hsv':
        temp = tf.image.rgb_to_hsv(image)
        return temp

    elif mode == 'yiq':
        temp = tf.image.rgb_to_yiq(image)
        return temp

    elif mode == 'yuv':
        temp = tf.image.rgb_to_yuv(image)
        return temp


def change_contrast(image, lower, upper, copies=1):
    copies = [tf.image.random_contrast(image, lower=lower, upper=upper) for _ in range(copies)]
    return copies


def change_brightness(image, delta, copies=1):
    copies = [tf.image.random_brightness(image, max_delta=delta) for _ in range(copies)]
    return copies


def change_hue(image, delta, copies=1):
    copies = [tf.image.random_hue(image, max_delta=delta) for _ in range(copies)]
    return copies


def gamma_transformation(image, gamma=0.3, copies=1):
    low = 1 - gamma
    up = 1 + gamma
    copies = [tf.image.adjust_gamma(image, gamma=np.random.uniform(low, up, 1)) for _ in range(copies)]
    return copies


def resize(image, size):
    return tf.image.resize(image, size)


def resize_smallest_side_different_scales(image, smallest_side_to=(128, 256, 384)):
    height, width = to_array(image).shape[:2]
    scaled_list = []

    if height < width:

        for scale in smallest_side_to:
            scaled = tf.image.resize(image, (scale, width))
            scaled_list.append(scaled)

        return scaled_list

    else:

        for scale in smallest_side_to:
            scaled = tf.image.resize(image, (height, scale))
            scaled_list.append(scaled)

        return scaled_list


def resize_with_aspect_ratio(image, target_width=(128, 256, 512), input_shape=(224, 224)):
    h, w = to_array(image).shape[:2]
    r = h / w
    resized = []

    for width in target_width:
        resized_h = int(r * width)
        resized_img = resize(image, (resized_h, width))
        resized.append(to_image(tf.image.resize_with_crop_or_pad(resized_img, input_shape[0], input_shape[1])))

    return resized


def bounding_boxes(offsets, dim):
    boxes = []

    for i in offsets:
        offset_height, offset_width = i
        target_height, target_width = dim
        boxes.append([offset_height, offset_width, target_height, target_width])

    return boxes


def random_sectioning(image, offsets, dims):
    boxes = bounding_boxes(offsets, dims)
    image_sections = []
    height, width = to_array(image).shape[:2]

    if (height < height // 2 + dims[0]) and (width < width // 2 + dims[1]):
        image = tf.image.resize(image, (dims[0] * 2, dims[1] * 2))

    if (height > height // 2 + dims[0]) and (width < width // 2 + dims[1]):
        image = tf.image.resize(image, (height, dims[1] * 2))

    if (height < height // 2 + dims[0]) and (width > width // 2 + dims[1]):
        image = tf.image.resize(image, (dims[0] * 2, width))

    for box in boxes:
        if random.choice([True, False]):
            section = tf.image.crop_to_bounding_box(image, box[0], box[1], box[2], box[3])
            image_sections.append(section)

    return image_sections


def aggressive_cropping(image, copies, crop_window, resize_smallest_side=None, output_shape=(224, 224)):
    global resized_copies

    if resize_smallest_side is not None:
        if isinstance(resize_smallest_side, int):
            img = resize(image, (resize_smallest_side, resize_smallest_side))

        if isinstance(resize_smallest_side, (list, tuple)):
            resized_copies = [tf.image.resize(image, (size, size)) for size in resize_smallest_side]

    if isinstance(crop_window, int):
        crops = [tf.image.random_crop(image, (crop_window, crop_window)) for _ in range(copies)]

        return [resize(crop_img, output_shape) for crop_img in crops]

    elif isinstance(crop_window, (list, tuple)):
        crops = [tf.image.random_crop(image, crop_window) for _ in range(copies)]

        return [resize(crop_img, output_shape) for crop_img in crops]


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


def cropping(image, filename=None, path=None, dst=None, central=True, random=False, fraction_low=0.5, fraction_high=0.9,
             random_width=224, random_height=224, copies=3, save_original=True):
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
    if path is not None:
        images = os.listdir(path)

        for image in images:
            image_path = os.path.join(path, image)
            img = img_to_array(PIL.Image.open(image_path))

            if central:
                for i in range(copies):
                    crop_area = np.round(np.random.uniform(fraction_low, fraction_high), 2)
                    cc_img = tf.image.central_crop(img, central_fraction=crop_area)

                    if dst is not None:
                        array_to_img(cc_img).save(os.path.join(dst, f'cc_img_{i}_{image}'))

                    else:
                        return array_to_img(cc_img)

            elif random:
                for i in range(copies):
                    rc_img = tf.image.random_crop(img, size=[random_width, random_height, 3])

                    if dst is not None:
                        array_to_img(rc_img).save(os.path.join(dst, f'rc_img_{i}_{image}'))

                    else:
                        return array_to_img(rc_img)

            if save_original:
                array_to_img(img).save(os.path.join(dst, image))

    if image:
        if central:
            for i in range(copies):
                crop_area = np.round(np.random.uniform(fraction_low, fraction_high), 2)
                cc_img = tf.image.central_crop(img_to_array(image), central_fraction=crop_area)

                if dst is not None:
                    array_to_img(cc_img).save(os.path.join(dst, f'cc_img_{i}_{filename}'))

                else:
                    return array_to_img(cc_img)

        elif random:
            for i in range(copies):
                rc_img = tf.image.random_crop(img_to_array(image),
                                              size=[random_width, random_height, 3])

                if dst is not None:
                    array_to_img(rc_img).save(os.path.join(dst, f'rc_img_{i}_{filename}'))

                else:
                    return array_to_img(rc_img)


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


def pipeline(file_name, src, dst, label):
    processed = []
    image = load_image(os.path.join(src, file_name))
    height, width = to_array(image)

    sections = random_sectioning(to_array(image),
                                 [[0, 0], [height // 2, 0], [0, width // 2], [height // 2, width // 2],
                                  [height // 4, width // 4]],
                                 [224, 224])
    resize_small_side = resize_smallest_side_different_scales(image, (224, 256, 384))
    resized_with_aspect_ratio = resize_with_aspect_ratio(image)
    resized_original = tf.image.resize_with_pad(image, 224, 224)

    for i, arr in enumerate(sections):
        filename = f'r-sec-{i}-{file_name}'
        processed.append([filename, label])
        to_image(arr).save(os.path.join(dst, filename))

    for i, arr in enumerate(resized_with_aspect_ratio):
        filename = f'r-to-ar-{i}-{file_name}'
        processed.append([filename, label])
        to_image(arr).save(os.path.join(dst, filename))

    for img in resize_small_side:
        rand_crop = aggressive_cropping(to_image(img), 4, (224, 224))

        for i, arr in enumerate(rand_crop):
            filename = f'agr-crop-{i}-{file_name}'
            processed.append([filename, label])
            to_image(arr).save(os.path.join(dst, filename))

    to_image(resized_original).save(os.path.join(dst, file_name))
