import numpy as np
from torch.utils.data import Dataset
import json
import sys
sys.path.append('../')
from dataset import partition_dataset 
import torch

class Clinic(object):
    def __init__(self, cfg):
        super(Clinic, self).__init__()
     
        self.data_path = '../dataset/5.3-lurq-update.xlsx'
        self.ratio = cfg.TRAINVAL_RATIO
        self.seed = cfg.RANDOM_SEED
   
        feats, targets, (trainval_mask, train_mask, \
                         val_mask, test_mask) = \
             partition_dataset(self.data_path, cfg.TARGET_NAME, 
                               cfg.USE_ICON, cfg.TRAINVAL_RATIO, 
                               cfg.RANDOM_SEED)        

        self.trainval = []
        self.train = []
        self.val = []
        self.test = []
           
        for idx in trainval_mask:
            self.trainval.append((feats[idx], targets[idx]))
 
        for idx in train_mask:
            self.train.append((feats[idx], targets[idx]))

        for idx in val_mask:
            self.val.append((feats[idx], targets[idx]))

        for idx in test_mask:
            self.test.append((feats[idx], targets[idx]))    

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
