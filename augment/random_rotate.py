import random
from PIL import Image


def random_rotate(img, label):
    angle = random.random()
    return img, label