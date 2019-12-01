import torch.nn as nn


def build_loss(cfg):
    loss_type = cfg.LOSS_TYPE
    assert loss_type in ['SmoothL1', 'MSE']
    
    if loss_type == 'SmoothL1':
        loss_func = nn.SmoothL1Loss(reduction='mean')
    elif loss_type == 'MSE':
        loss_func = nn.MSELoss(reduction='mean')

    return loss_func
