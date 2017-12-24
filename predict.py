import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import cv2
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
import models
import utils
import data
import random


def main():    
    use_cuda = torch.cuda.is_available()
    path = os.path.expanduser('~/codedata/seg/')
    
    dataset = data.VOCClassSeg(root=path,
                            split='val.txt',
                            transform=True)

    model = models.FCN8(path)
    model.load('SBD.pth')
    model.eval()
    
    if use_cuda:
        model.cuda()
    
    criterion = utils.CrossEntropyLoss2d(size_average=False, ignore_index=255)

    for i in range(len(dataset)):
        idx = random.randrange(0, len(dataset))
        img, label = dataset[idx]
        img_name = str(i)

        img_src, _ = dataset.untransform(img, label)
        cv2.imwrite(path + 'image/%s_src.jpg' % img_name, img_src)
        utils.tool.labelTopng(label, path + 'image/%s_label.png' % img_name)
        
        print(img_name)

        if use_cuda:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img.unsqueeze(0), volatile=True)
        label = Variable(label.unsqueeze(0), volatile=True)

        out = model(img)
        loss = criterion(out, label)
        print('loss:', loss.data[0])

        label = out.data.max(1)[1].squeeze_(1).squeeze_(0)
        if use_cuda:
            label = label.cpu()
        
        utils.tool.labelTopng(label, path + 'image/%s_out.png' % img_name)
        
        if i == 10:
            break

if __name__ == '__main__':
    main()
    