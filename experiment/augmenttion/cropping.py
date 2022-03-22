import random
import tensorflow as tf


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