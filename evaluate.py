import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import models
import data
import utils
import numpy as np
from torch.autograd import Variable

def evaluate():
    use_cuda = torch.cuda.is_available()
    path = os.path.expanduser('~/codedata/seg/')
    val_data = data.VOCClassSeg(root=path,
                            split='val.txt',
                            transform=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=5)  
    print('load model .....') 
    model = models.FCN8(path)
    model.load('SBD.pth')
    if use_cuda:
        model.cuda()
    model.eval()

    label_trues, label_preds = [], []
    # for idx, (img, label) in enumerate(val_loader):
    for idx in range(len(val_data)):
        img, label = val_data[idx]
        img = img.unsqueeze(0)
        if use_cuda:
            img = img.cuda()
        img = Variable(img, volatile=True)
        
        out = model(img)
        pred = out.data.max(1)[1].squeeze_(1).squeeze_(0)
        
        if use_cuda:
            pred = pred.cpu()
        label_trues.append(label.numpy())
        label_preds.append(pred.numpy())

        if idx % 30 == 0:
            print('evaluate [%d/%d]' % (idx, len(val_loader)))
    
    metrics = utils.tool.accuracy_score(label_trues, label_preds)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
            Accuracy: {0}
            Accuracy Class: {1}
            Mean IU: {2}
            FWAV Accuracy: {3}'''.format(*metrics))

if __name__ == '__main__':
    evaluate()