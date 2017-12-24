import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision
import models
import data
import utils

batch_size = 1
learning_rate = 1e-10
epoch_num = 30
best_test_loss = np.inf
pretrained = 'reload'
use_cuda = torch.cuda.is_available()
path = os.path.expanduser('~/codedata/seg/')

print('load data....')
train_data = data.SBDClassSeg(root=path,
                            split='train.txt',
                            transform=True)
train_loader = torch.utils.data.DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=5)
val_data = data.VOCClassSeg(root=path,
                            split='val_val.txt',
                            transform=True)
val_loader = torch.utils.data.DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=5)

print('load model.....')
model = models.FCN8(path)
# model = models.FCN8(path)

if pretrained is 'pretrain':
    VGG16 = torchvision.models.vgg16(pretrained=True)
    model.copy_params_from_vgg16(VGG16)
elif pretrained is 'reload':
    model.load('SBD.pth')
else:
    print("no pretrained model load")

if use_cuda:
    model.cuda()

criterion = utils.loss.CrossEntropyLoss2d(size_average=False, 
                                        ignore_index=255)
optimizer = torch.optim.SGD([{'params': models.get_parameters(model, bias=False)},
                            {'params': models.get_parameters(model, bias=True),
                            'lr':learning_rate*2, 'weight_decay': 0}],
                            lr=learning_rate,
                            momentum=0.99,
                            weight_decay=5e-4)
                      
vis = utils.Visualizer()

print('begin to train....')
def train(epoch):
    model.train()
    total_loss = 0.
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        N = imgs.size(0)
        if use_cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()
        imgs = Variable(imgs)
        labels = Variable(labels)

        out = model(imgs)
        loss = criterion(out, labels)
        loss /= N
        print('loss', loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]  # return float
        if (batch_idx+1) % 20 == 0:
            print('train epoch [%d/%d], iter[%d/%d], lf %.5f, aver_loss %.5f' % (epoch, 
                    epoch_num, batch_idx, len(train_loader), learning_rate, total_loss/(batch_idx+1)))

        if (batch_idx+1) % 30 == 0:
            vis.plot_train_val(loss_train=total_loss/(batch_idx+1))
        
        # if batch_idx == 22:
        #     break
        assert total_loss is not np.nan
        assert total_loss is not np.inf

    total_loss /= len(train_loader)
    print('train epoch [%d/%d] average_loss %.5f' % (epoch, epoch_num, total_loss))


def test(epoch):
    model.eval()
    total_loss = 0.
    for batch_idx, (imgs, labels) in enumerate(val_loader):
        N = imgs.size(0)
        if use_cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()
        imgs = Variable(imgs, volatile=True)
        labels = Variable(labels, volatile=True)
        out = model(imgs)
        loss = criterion(out, labels)
        loss /= N
        total_loss += loss.data[0]

        if (batch_idx+1) % 3 == 0:
            print('test epoch [%d/%d], iter[%d/%d], aver_loss %.5f' % (epoch, 
                    epoch_num, batch_idx, len(val_loader), total_loss/(batch_idx+1)))
    
    total_loss /= len(val_loader)
    vis.plot_train_val(loss_val=total_loss)
    print('test epoch [%d/%d] average_loss %.5f' % (epoch, epoch_num, total_loss))

    global best_test_loss
    if best_test_loss > total_loss:
        best_test_loss = total_loss
        print('best loss....')
        model.save('SBD.pth')

if __name__ == '__main__':
    for epoch in range(epoch_num):
        train(epoch)
        test(epoch)

        # adjust learning rate
        if epoch == 1 or epoch == 2:
            learning_rate *= 0.1
            optimizer.param_groups[0]['lr'] = learning_rate
            optimizer.param_groups[1]['lr'] = learning_rate * 2

