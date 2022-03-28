import random
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img


def load_img_to_array(path):
    return img_to_array(load_img(path))


def resize(image, size):
    return tf.image.resize(image, size)


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

    for box in boxes:
        if random.choice([True, False]):
            section = tf.image.crop_to_bounding_box(image, box[0], box[1], box[2], box[3])
            image_sections.append(section)

    return image_sections


def aggressive_cropping(image, copies, crop_window, resize_smallest_side, output_shape):
    global img, resized_copies, crops

    if isinstance(resize_smallest_side, int):
        img = resize(image, (resize_smallest_side, resize_smallest_side))

    if isinstance(resize_smallest_side, (list, tuple)):
        resized_copies = [tf.image.resize(image, (size, size)) for size in resize_smallest_side]

    if isinstance(crop_window, int):
        if isinstance(resize_smallest_side, int):
            crops = [tf.image.random_crop(img, crop_window) for _ in range(copies)]
        elif isinstance(resize_smallest_side, (list, tuple)):
            crops = [tf.image.random_crop(img_, crop_window) for _ in range(copies) for img_ in
                     resized_copies]

    elif isinstance(crop_window, (list, tuple)):
        if isinstance(resize_smallest_side, int):
            crops = [tf.image.random_crop(img, crop_window) for _ in range(copies)]
        elif isinstance(resize_smallest_side, (list, tuple)):
            crops = [tf.image.random_crop(img_, crop_window) for _ in range(copies) for img_ in resized_copies]

    return [resize(crop_img, output_shape) for crop_img in crops]
