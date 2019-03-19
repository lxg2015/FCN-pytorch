import os
import torch
import cv2
import random
import scipy.io
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms

from torch.utils import data
from data.voc import load_voc
from augment import *


mean_rgb = [0.485, 0.456, 0.406]
std_rgb = [0.229, 0.224, 0.225]
normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_rgb, std=std_rgb)
])


def train_transform(x, y, img_size):
    x, y = random_flip(x, y)
    x, y = random_crop(x, y)
    x, y = resize(x, y, size=img_size)
    x = normalize(x)
    y = np.array(y)
    return x, y


def test_transform(x, y, img_size):
    x, y = resize(x, y, size=img_size)
    x = normalize(x)
    y = np.array(y)
    return x, y


class SegDataset(data.Dataset):
    def __init__(self, dataset='voc', split='train', img_size=512):
        if dataset == 'voc':
            data, label = load_voc(split)
        else:
            raise "have not define %s" % dataset
        
        self.data = data
        self.label = label
        self.img_size = img_size

        if split == 'train':
            self.transform = train_transform
        elif split == 'test' or split == 'val':
            self.transform = test_transform
        else:
            self.transform = normalize

    def __getitem__(self, idx):
        data_file = self.data[idx]
        label_file = self.label[idx]

        # load img and label
        # cv2.imread(file, 0), param 0 restrict to read gray not convert to bgr
        # however here label.png is not just gray, if read by cv2.imread(file,0),
        # result is array([  0,  52, 132, 147, 220], dtype=uint8)
        # while PIL read array([  0,   5,  11,  15, 255], dtype=uint8)
        # which is our expected
        img = Image.open(data_file)
        label = Image.open(label_file)
        if self.transform is not None:
            img, label = self.transform(img, label, self.img_size)

        return img, label
    
    def __len__(self):
        return len(self.data)

    def untransform(self, img, label=None):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img = img * std_rgb + mean_rgb
        img *= 255
        img = img.astype(np.uint8)  # rgb
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        label = None if label is None else label.numpy()
        return img, label