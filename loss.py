import torch.nn as nn
import torch


def build_loss(cfg):
    loss_type = cfg.LOSS_TYPE
    
    if loss_type == 'SmoothL1':
        loss_func = nn.SmoothL1Loss(reduction='mean')
    elif loss_type == 'MSE':
        loss_func = nn.MSELoss(reduction='mean')
    elif loss_type == 'CE':
        loss_func = nn.BCELoss(reduction='mean')
    else:
        raise Exception('Invalid loss type:', loss_type)

    def loss_fn(score, target):
        if loss_type == 'CE':
            score = torch.sigmoid(score)
        loss = loss_func(score, target)
        return loss

    return loss_fn
