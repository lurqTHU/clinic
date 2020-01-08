import numpy as np
from torch.utils.data import Dataset
import json
from data.utils import construct_Clinic
import torch

class Clinic(object):
    def __init__(self, cfg):
        super(Clinic, self).__init__()
     
        self.data_path = '/space1/home/lurq/code/clinic/dataset/update.xlsx'
        self.ratio = cfg.TRAIN_RATIO
        self.seed = cfg.RANDOM_SEED
    
        np.random.seed(self.seed)
        
        self.train = []
        self.val = []
        infos = construct_Clinic(self.data_path)
        feats = infos['feat']
        targets = infos['target']

        print('Feature dimension: ', feats.shape[1])
            
        total = feats.shape[0]
        rand_uniform = np.random.uniform(0, 1, total)
        train_mask = np.where(rand_uniform < self.ratio)[0]
        val_mask = np.where(rand_uniform >= self.ratio)[0]
          
        print('Train mask:', train_mask)
        print('Val mask:', val_mask)
        
        for idx in train_mask:
            self.train.append((feats[idx], targets[idx]))

        for idx in val_mask:
            self.val.append((feats[idx], targets[idx]))     
    
    
class ImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, item):
        feat, target = self.dataset[item]
        feat = torch.tensor(feat, dtype=torch.float32)
        return feat, target

    def __len__(self):
        return len(self.dataset)
    
if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from config import cfg
    dataset = Clinic(cfg)
    iter_dataset = ImageDataset(dataset)
    import pdb;pdb.set_trace()
