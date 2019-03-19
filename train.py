import os
import torch
import models
import data
import utils
import torchvision

import numpy as np
import torch.nn as nn
from utils.cfg import cfg

os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpus']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = cfg['batch_size']
learning_rate = cfg['learning_rate']
epoch_num = cfg['epoches']
start_epoch = 0

print('load data....')
train_data = data.SegDataset(dataset=cfg['dataset'], split='train', img_size=cfg['img_size'])
val_data = data.SegDataset(dataset=cfg['dataset'], split='val', img_size=cfg['img_size'])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=cfg['num_workers'])
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=cfg['num_workers'])


print('load model.....')
if cfg['model'] == 'FCN':
    net = models.FCN(cfg['num_classes'], cfg['num_loss'], backbone=cfg['backbone']).to(device)
elif cfg['model'] == 'UNet':
    net = models.UNet(3, cfg['num_classes']).to(device)
elif cfg['model'] == 'PSPNet':
    net = models.PSPNet(cfg['num_classes']).to(device)
else:
    print('%s is not defined' % cfg['model'])

if cfg['pretrained']:
    print('load %s' % cfg['pretrained'])
    net.load_pretrained(cfg['pretrained'])

if cfg['resume']:
    print('resume from %s' % cfg['checkpoint'])
    state_dict = torch.load(cfg['checkpoint'])
    net.load_state_dict(state_dict['net'])
    start_epoch = state_dict['epoch']
    print('start from epoch %d' % start_epoch)
else:
    print("no pretrained model load")


criterion = models.SegLoss(ignore_index=255)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.99, weight_decay=cfg['weight_decay'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['step_lr'], gamma=0.1)
vis = utils.Visualizer()
for i in range(start_epoch):
    scheduler.step()
print('begin to train....')


def train(epoch):
    net.train()
    total_loss = 0.
    torch.set_grad_enabled(True)
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        out = net(imgs)

        loss = criterion(out, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # return float
        print('epoch [%d/%d], iter[%d/%d], lr %.2f, loss: %.2f, aver_loss %.5f' %
              (epoch, epoch_num, batch_idx, len(train_loader), learning_rate, loss.item(),
               total_loss / (batch_idx + 1)))
        vis.plot_train_val(loss_train=total_loss / (batch_idx + 1))

    total_loss /= len(train_loader)
    print('train epoch [%d/%d] average_loss %.5f' % (epoch, epoch_num, total_loss))

    def savemodel(name='ckpt_best.pth'):
        print('Saving %s.. \ntest_loss:%f' % (name, total_loss))
        state = {
            'net': net.state_dict(),
            'loss': total_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('./checkpoint'):
            os.mkdir('./checkpoint')
        torch.save(state, './checkpoint/' + name)
    savemodel(name=cfg['model'] + '_test.pth')


def test(epoch):
    net.eval()
    total_loss = 0.
    torch.set_grad_enabled(False)
    for batch_idx, (imgs, labels) in enumerate(val_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        out = net(imgs)
        loss = criterion(out, labels.long())
        total_loss += loss.item()

        print('epoch [%d/%d], iter[%d/%d], aver_loss %.2f' %
              (epoch, epoch_num, batch_idx, len(val_loader), total_loss / (batch_idx + 1)))

    total_loss /= len(val_loader)
    vis.plot_train_val(loss_val=total_loss)
    print('test epoch [%d/%d] average_loss %.5f' % (epoch, epoch_num, total_loss))


if __name__ == '__main__':
    for epoch in range(start_epoch, epoch_num):
        train(epoch)
        test(epoch)
        scheduler.step()
