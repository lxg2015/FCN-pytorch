from .basicModule import BasicModule
import torch
import torch.nn as nn
import numpy as np

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsample_weight(in_channels, out_channels, kernel_size):
    '''
    make a 2D bilinear kernel suitable for upsampling
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]  # list (64 x 1), (1 x 64)
    filt = (1 - abs(og[0] - center) / factor) * \
            (1 - abs(og[1] - center) / factor)  # 64 x 64
    weight = np.zeros((in_channels, out_channels, kernel_size,
                     kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :,:] = filt
    
    return torch.from_numpy(weight).float()

class FCN32(BasicModule):
    def __init__(self, path, n_class=21):
        super(FCN32, self).__init__(path) # pascol voc 21 class
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(True)
        self.drop7 = nn.Dropout2d()

        # fr ?? score from; upscore ?? upsample score
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64,
                                    stride=32, 
                                    bias=False)
        self._init_weights()

    def forward(self, x):
        h = x
        
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        
        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        
        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        
        h = self.relu6(self.fc6(h)) # conv2d kernel=7
        h = self.drop6(h)

        h = self.relu7(self.fc7(h)) # conv2d kernel=1
        h = self.drop7(h)

        h = self.score_fr(h) # conv2d kernel=1
        h = self.upscore(h) # upsample
        
        # 32 means zoom out 32 times
        # fc6 = (image_size/32 - 7)/1 + 1 = (image_size-192)/32
        # after pad 100, why not 97?? 
        # fc6 = (image_size+6)/32
        # after upscore
        # upsize = (fc6-1)*32+64 = image_size+38
        h = h[:,:,19:19+x.size(2), 19:19+x.size(3)].contiguous()

        return h
    
    def _init_weights(self):
        '''
        hide method, used just in class
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                # if m.bias is not None:
                m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsample_weight(m.in_channels, 
                            m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight) # copy not = ?

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']): # 0,3 is nn.linear
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.data.size())
            l2.bias.data = l1.bias.data.view(l2.bias.data.size())
            