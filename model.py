import torch
from torch import nn


class BPnet(nn.Module):
    def __init__(self, num_layers, in_planes, mid_planes):
        super(BPnet, self).__init__()
        
        self.in_layer = nn.Linear(self.in_planes, self.mid_planes, bias=True)
        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Linear(self.mid_planes, self.mid_planes, bias=True)
        self.mid_layer = nn.Sequential(*layers)
        self.out_layer = nn.Linear(self.mid_planes, 1, bias=True)
     
      
    def forward(self, x):
        out = self.in_layer(x)
        out = self.mid_layer(out)
        out = self.out_layer
        return out
        
    
