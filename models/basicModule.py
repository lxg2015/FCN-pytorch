import torch
import time


class BasicModule(torch.nn.Module):
    def __init__(self, path):
        super(BasicModule, self).__init__()
        self.model_name = type(self).__name__  # model name
        self.path = path
    
    def load(self, name=None):
        full_name = self.model_name + '_' + name
        self.load_state_dict(torch.load(self.path + full_name))
        print('load %s successfuly....' % full_name)
    
    def save(self, name=None):
        if name == None:
            full_name = time.strftime(self.model_name + '_' + '%m%d_%H:%M:%S.pth')
        else:
            full_name = self.model_name + '_' + name 
        torch.save(self.state_dict(), self.path + full_name)
        print('save %s successfuly....' % full_name)
