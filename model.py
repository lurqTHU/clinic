import torch
from torch import nn


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0.0)


activation_dict = {'relu': nn.ReLU, 
                   'sigmoid': nn.Sigmoid,
                   'tanh': nn.Tanh}


class BPnet(nn.Module):
    def __init__(self, num_layers, in_planes, 
                 mid_planes, activation_type='relu'):
        super(BPnet, self).__init__()
       
        self.num_layers = num_layers
        self.in_planes = in_planes
        self.mid_planes = mid_planes
        self.activation = activation_dict[activation_type]()
        self.dropout = nn.Dropout(p=0.6)
         
        self.in_layer = nn.Linear(self.in_planes, self.mid_planes, bias=True)
        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Linear(self.mid_planes, self.mid_planes, bias=True))
            layers.append(self.activation)
            layers.append(self.dropout)
        self.mid_layer = nn.Sequential(*layers)
        self.out_layer = nn.Linear(self.mid_planes, 1, bias=True)
     
        self.apply(weights_init_xavier)
      
    def forward(self, x):
        out = self.in_layer(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.mid_layer(out)
        out = self.out_layer(out)
        return out
  
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])          
