'''resnet50_v1d glouncv'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck_v1b(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, expansion=4):
        super(Bottleneck_v1b, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
    
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True),
                nn.Conv2d(in_planes, self.expansion * planes, 1, 1, padding=0, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        return F.relu(out)


class ResNet_v1d(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet_v1d, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1, padding=1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.squeeze()
        x = self.fc(x)
        x = x.sigmoid()
        return x
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def load_dict(self, path):
        import mxnet
        state = mxnet.ndarray.load(path)
        keys = list(state.keys())
        for key in keys:
            if 'gamma' in key:
                tmp = key.replace('gamma', 'weight')
                state[tmp] = state.pop(key)
            if 'beta' in key:
                tmp = key.replace('beta', 'bias')
                state[tmp] = state.pop(key)

        keys = list(self.state_dict().keys())
        for key in keys:
            if 'num_batches' in key:
                continue
            for s1, s2 in zip(state[key].shape, self.state_dict()[key].shape):
                assert s1 == s2, "%s not equal" % key

            self.state_dict()[key].copy_(torch.from_numpy(state[key].asnumpy()))


class FPN_v1d(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN_v1d, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1, padding=1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top-down layers
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # top down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5
    
    def getfeature(self):
        return self.feature

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y


def FPN50_v1d():
    return FPN_v1d(Bottleneck_v1b, [3, 4, 6, 3])


def ResNet50_v1d():
    return ResNet_v1d(Bottleneck_v1b, [3, 4, 6, 3])

if __name__ == '__main__':
    import torchvision.transforms as transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    net = ResNet_v1d(block=Bottleneck_v1b, num_blocks=[3, 4, 6, 3])
    net.load_dict(path='/home/lxg/.torch/models/resnet50_v1d-117a384e.params')
    torch.save(net.state_dict(), '/home/lxg/.torch/models/resnet50_v1d_from_glouncv.pth')
    img = Image.open('/home/lxg/data/dog.jpeg').resize((224, 224))
    img = transform(img)
    out = net(img.unsqueeze(0))
    print(out.argmax())
    print(out[out.argmax()])
