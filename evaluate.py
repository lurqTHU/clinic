import numpy as np
import torch


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Acc(object):
    def __init__(self, thres=0.1, metric='L1'):
        super(Acc, self).__init__()
        self.thres = thres
        self.metric = metric
        self.results = []
        self.targets = []

    def reset(self):
        self.results = []
        self.targets = []

    def update(self, data):
        result, target = data
        self.results.append(result)
        self.targets.extend(np.asarray(target))

    def compute(self):
        results = torch.cat(self.results, dim=0).cpu().numpy()
        targets = np.asarray(self.targets)
        
        if self.metric == 'L1':
            dist = np.abs(results-targets)
        elif self.metric == 'L2':
            dist = np.sqrt((results-targets) * (results-targets))
        elif self.metric == 'PROB':
            dist = 1.0/(1+np.exp(-results))
            dist = np.where(targets == 1, dist, 1-dist)
        
        if self.metric in ['L1', 'L2']:
            acc = np.sum(dist <= self.thres) / dist.shape[0]
            print('Mean distance:', np.mean(dist), 'Accuracy:', acc)
        elif self.metric == 'PROB':
            acc = np.sum(dist > self.thres) / dist.shape[0]
            loss = np.mean(-np.log2(dist))
            print('Accuracy at threshold {}:'.format(self.thres), acc, 'Loss:', loss)


