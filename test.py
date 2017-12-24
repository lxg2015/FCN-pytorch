import os
import models
import torch
import data
import numpy as np
import random
import utils
import torchvision.transforms as transforms
import cv2
import PIL.Image as Image

def testFCN32():
    from torch.autograd import Variable
    model = models.FCN32()
    data = Variable(torch.randn(1,3,512,512))
    out = model(data)
    print('input', data.size(), 'out', out.size())

def testVOC():
    dataset = data.VOCClassSeg(root=path,
                            split='train.txt',
                            transform=True)
    idx = random.randrange(0, 20)
    image, label = dataset[idx]
    image, label = dataset.untransform(image, label)

    label_pil = utils.tool.colorize_mask(label)
    label_pil.show()

    cv2.imshow('image', image)
    cv2.waitKey(0)

def testSBD():
    dataset = data.SBDClassSeg(root=path,
                            split='train.txt',
                            transform=True)
    idx = random.randrange(0, len(dataset))
    image, label = dataset[idx]
    image, label = dataset.untransform(image, label)
    
    print('label', np.unique(label))

    label_pil = utils.tool.colorize_mask(label)
    label_pil.show()

    cv2.imshow('image', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    path = os.path.expanduser('~/codedata/seg/')
    # testFCN32()
    # testVOC()
    testSBD()