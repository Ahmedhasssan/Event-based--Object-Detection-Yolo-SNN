import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import IPython
import torch.nn.init as init
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional, cast
from torch import Tensor
from collections import OrderedDict 

from modules import *
#from PACT import *

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1=nn.Conv2d(1, 16, 3, 1)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(2, 2)
        self.conv2=nn.Conv2d(16, 32, 3, 1)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(2, 2)
        self.linear1=nn.Linear(900*32, 500)
        self.relu3=nn.ReLU()
        self.linear2=nn.Linear(500, 10)
    def forward(self, x):
        x=self.conv1(x)
        import pdb;pdb.set_trace()
        x=self.relu1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        x=nn.Flatten(x)
        x=self.linear1(x)
        x=self.relu3(x)
        x=self.linear2(x)
        x=nn.LogSoftmax(x)
        return x
        
class QUAutoencoder(nn.Module):
    def __init__(self):
        super(QUAutoencoder,self).__init__()
        self.encoder = nn.Sequential(
            QConv2d(2, 16, 3, stride=1, padding=1, wbit=4, abit=32),
            nn.ReLU(),
            #nn.MaxPool2d(2,2),
            QConv2d(16, 16, 3, stride=1, padding=1, wbit=4, abit=4),
            #nn.ReLU(),
            #nn.MaxPool2d(2,2),
            )
            
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
            )
    def forward(self, x):
        x = self.encoder(x)
        x = torch.sigmoid(x)
        encoded = x
        x = torch.logit(x, eps=0.001)
        x = self.decoder(x)
        return x
        
    def get_group_mp(self):
        self.ch_group=1
        val = torch.Tensor()
        if torch.cuda.is_available():
            val = val.cuda()

        count = 0
        for m in self.modules():
            if isinstance(m, QConv2d):
                kw = m.weight.size(2)
                if kw != 1:
                    if not count in [0]:
                        w_l = m.weight
                        num_group = w_l.size(0) * w_l.size(1) // self.ch_group
                        w_l = w_l.view(w_l.size(0), w_l.size(1) // self.ch_group, self.ch_group, kw, kw)
                        w_l = w_l.contiguous().view((num_group, self.ch_group*kw*kw))

                        g = w_l.abs().mean(dim=1)
                        val = torch.cat((val.view(-1), g.view(-1)))
                    count += 1
        return val  
          
    def get_global_mp_thre(self, ratio):
        grp_val = self.get_group_mp()
        sorted_block_values, indices = torch.sort(grp_val.contiguous().view(-1))
        thre_index = int(grp_val.data.numel() * ratio)
        threshold = sorted_block_values[thre_index]
        return threshold
        
class NewModel(nn.Module):
    def __init__(self, output_layers, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        #print(self.output_layers)
        self.selected_out = OrderedDict()
        #PRETRAINED MODEL
        self.pretrained = QUAutoencoder()
        self.fhooks = []

        for i,l in enumerate(list(self.pretrained._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.pretrained,l).register_forward_hook(self.forward_hook(l)))
    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out 