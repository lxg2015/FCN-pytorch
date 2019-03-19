import numpy as np


def load_voc(split):
    dataset_dir = '/home/we/dataset/voc/VOCdevkit/VOC2012/'
    imgsets_file = "/home/we/dataset/voc/VOCdevkit/VOC2012/ImageSets/Segmentation/%s.txt" % split

    with open(imgsets_file) as f:
        imglist = f.readlines()

    data, label = [], []
    for img in imglist:
        img = img.strip()
        data.append(dataset_dir + 'JPEGImages/%s.jpg' % img)
        label.append(dataset_dir + 'SegmentationClass/%s.png' % img)

    return data, label


class_names = np.array([
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'potted plant',
    'sheep',
    'sofa',
    'train',
    'tv/monitor',
])
