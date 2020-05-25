from bisect import bisect_right
import torch
import torch.optim
import torch.optim.lr_scheduler as scheduler

def build_optimizer(cfg, model):
    
    base_learning_rate = cfg.BASE_LR
    learning_rate_weight_decay = cfg.WEIGHT_DECAY
    learning_rate_bias = cfg.BIAS_LR_FACTOR
    model_optimizer = cfg.OPTIMIZER_NAME
    
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_learning_rate
        weight_decay = learning_rate_weight_decay
        if 'bias' in key:
            lr = base_learning_rate * learning_rate_bias
        params += [{'params': [value], 'lr':lr, 'weight_decay': weight_decay}]
    
    if model_optimizer == 'SGD':
        optimizer = getattr(torch.optim, model_optimizer)(params, 
                            momentum=0.9)
    else:
        optimizer = getattr(torch.optim, model_optimizer)(params)

    return optimizer


class MultiStepLr(scheduler._LRScheduler):
    def __init__(
              self, 
              optimizer, 
              milestones,
              gamma=0.1,
              last_epoch=-1
    ):
        if not list(milestones) == sorted(milestones):
            raise Exception(
                'Milestones should be a list of'
                ' increasing integers. Got {}',
                milestones
            )
        
        self.milestones = milestones
        self.gamma = gamma
        super(MultiStepLr, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch) 
            for base_lr in self.base_lrs
        ]
