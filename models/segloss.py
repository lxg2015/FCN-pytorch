import torch
import torch.nn as nn


class SegLoss(nn.Module):
    def __init__(self, ignore_index):
        super(SegLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=255)
        # nn.BCELoss()
    
    def forward(self, xs, y):
        batch_size = xs[0].shape[0]
        loss = 0
        for x in xs:
            loss += self.criterion(x, y)
        loss /= (batch_size * len(xs))
        return loss
        