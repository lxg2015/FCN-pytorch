import random
from PIL import Image


def resize(img, label, size, random_interpolation=True):
    ow = oh = size
    method = random.choice([
        Image.NEAREST,
        Image.HAMMING,
        Image.BICUBIC,
        Image.LANCZOS,
        Image.BILINEAR
    ]) if random_interpolation else Image.BILINEAR
    img = img.resize((ow, oh), method)
    label = label.resize((ow, oh), Image.NEAREST)
    return img, label
    