import random
from PIL import Image


def random_flip(img, label):
    '''Randomly flip PIL image.

    Args:
      img: (PIL.Image) image to be flipped.

    Returns:
      img: (PIL.Image) randomly flipped image.
    '''
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)

    return img, label
    