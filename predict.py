import os
import torch
import cv2
import models
import utils
import data
import random

import numpy as np
import torchvision.transforms as transforms
from utils.cfg import cfg

os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpus']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = data.SegDataset(dataset=cfg['dataset'], split='val')

print('load model.....')
if cfg['model'] == 'FCN':
    net = models.FCN(cfg['num_classes'], cfg['num_loss'], backbone=cfg['backbone']).to(device)
elif cfg['model'] == 'UNet':
    net = models.UNet(3, cfg['num_classes']).to(device)
elif cfg['model'] == 'PSPNet':
    net = models.PSPNet(cfg['num_classes']).to(device)
else:
    print('%s is not defined' % cfg['model'])

net.load_state_dict(torch.load(cfg['checkpoint'])['net'])
net.eval()
torch.set_grad_enabled(False)

for i in range(len(dataset)):
    img, label = dataset[i]
    img_src, _ = dataset.untransform(img)
    cv2.imwrite('./image/%d_src.jpg' % i, img_src)

    img = img.to(device)
    out = net(img.unsqueeze(0))
    y0 = out[0].data.cpu()
    label = y0.max(1)[1].squeeze_(1).squeeze_(0)
    utils.tool.labelTopng(label, './image/%d_predict.png' % i)

    print(i)
    if i > 10:
        break
