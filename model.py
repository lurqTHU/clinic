import torch
from torch import nn


def weights_init_linear(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class BPnet(nn.Module):
    def __init__(self, num_layers, in_planes, mid_planes):
        super(BPnet, self).__init__()
       
        self.num_layers = num_layers
        self.in_planes = in_planes
        self.mid_planes = mid_planes
         
        self.in_layer = nn.Linear(self.in_planes, self.mid_planes, bias=True)
        self.relu_in = nn.ReLU(inplace=True)
        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Linear(self.mid_planes, self.mid_planes, bias=True))
            layers.append(nn.ReLU(inplace=True))
        self.mid_layer = nn.Sequential(*layers)
        self.out_layer = nn.Linear(self.mid_planes, 1, bias=True)
     
        self.apply(weights_init_linear)
      
    def forward(self, x):
        out = self.in_layer(x)
        out = self.relu_in(out)
        out = self.mid_layer(out)
        out = self.out_layer(out)
        return out
            
