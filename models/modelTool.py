import torch
import torch.nn as nn

def get_parameters(model, bias=False):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight can be frozen for bilinear upsampling
            if bias:
                assert m.bias is None
            else:
                yield m.weight
        else:
            print("module: %s 0 leraning rate" % str(m))