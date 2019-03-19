import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fpn import FPN18, FPN50
from models.gloun import FPN50_v1d


class FCN(nn.Module):
    def __init__(self, num_classes, num_layers, backbone='resnet18'):
        super(FCN, self).__init__()
        if backbone == 'resnet18':
            self.fpn = FPN18()
        elif backbone == 'resnet50':
            self.fpn = FPN50()
        elif backbone == 'resnet50_v1d':
            self.fpn = FPN50_v1d()
        else:
            print("%s not defined" % backbone)

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.head = nn.ModuleList()
        for _ in range(num_layers):
            self.head.append(self._make_head())

    def forward(self, x):
        _, _, h, w = x.shape
        fms = self.fpn(x)
        outs = []
        for i in range(self.num_layers):
            out = self.head[i](fms[i])
            out = F.interpolate(out, (h, w), mode='bilinear')
            outs.append(out)
        return outs

    def _make_head(self):
        return nn.Sequential(
            nn.Conv2d(256, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(64, self.num_classes, 1, 1, 1)
        )
    
    def load_pretrained(self, path):
        self.fpn.load_state_dict(torch.load(path), strict=False)
