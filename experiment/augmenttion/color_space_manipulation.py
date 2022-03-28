import numpy as np
import tensorflow as tf


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
