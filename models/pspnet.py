import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gloun import ResNet50_v1d
from models.fpn import ResNet18, ResNet50
# Pyramid Scene Parsing Network
# concate feature map and predict


class _PyramidPooling(nn.Module):
    def __init__(self, inplanes):
        super(_PyramidPooling, self).__init__()
        planes = inplanes // 4
        self.conv1 = self._make_1x1conv(inplanes, planes)
        self.conv2 = self._make_1x1conv(inplanes, planes)
        self.conv3 = self._make_1x1conv(inplanes, planes)
        self.conv4 = self._make_1x1conv(inplanes, planes)

    def upsample(self, x, h, w):
        return F.interpolate(x, (h, w), mode='bilinear')
    
    def pool(self, x, size):
        return F.adaptive_avg_pool2d(x, output_size=size)

    def forward(self, x, h, w):
        # _, _, h, w = x.shape
        x0 = self.upsample(x, h, w)
        x1 = self.upsample(self.conv1(self.pool(x, 1)), h, w)
        x2 = self.upsample(self.conv1(self.pool(x, 2)), h, w)
        x3 = self.upsample(self.conv1(self.pool(x, 3)), h, w)
        x4 = self.upsample(self.conv1(self.pool(x, 6)), h, w)
        return torch.cat([x0, x1, x2, x3, x4], dim=1)

    def _make_1x1conv(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )


class PSPNet(nn.Module):
    def __init__(self, num_classes, backbone='resnet18'):
        super(PSPNet, self).__init__()
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
        self.pyramid_pooling = _PyramidPooling(128)
        self.head = self._make_head(256)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)  # s4, 64
        x = self.layer1(x)  # s4, 64
        x = self.layer2(x)  # s8, 128
        # x = self.layer3(x)  # s16
        # x = self.layer4(x)  # s32

        x = self.pyramid_pooling(x, w, h)
        x = self.head(x)
        return [x, ]
    
    def _make_head(self, inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(inplanes // 4, self.num_classes, 1, 1)
        )
    
    def load_pretrained(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        