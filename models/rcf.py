import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gloun import ResNet50_v1d
from models.fpn import ResNet18, ResNet50
# Richer Convolutional Features for Edge Detection
# multi-layer loss


class RCFNet(nn.Module):
    def __init__(self, num_classes, backbone='resnet18'):
        super(RCFNet, self).__init__()
        if backbone == 'resnet18':
            model = ResNet18()
        elif backbone == 'resnet50':
            model = ResNet50()
        elif backbone == 'resnet50_v1d':
            model = ResNet50_v1d()
        else:
            print('%s not defined' % backbone)

        self.num_classes = num_classes

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.head = self._make_head(128)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x1 = self.maxpool(x)  # s4, 64
        x2 = self.layer1(x1)  # s4, 64
        x3 = self.layer2(x2)  # s8, 128
        x4 = self.layer3(x3)  # s16
        x5 = self.layer4(x4)  # s32

        xs = [x1, x2, ]
        xs_up = []
        for x in xs:
            xs_up.append(F.interpolate(x, (h, w), mode='bilinear'))
        x = torch.cat(xs_up, dim=1)
        x = self.head(x)
        return [x, ]
    
    def load_pretrained(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        
