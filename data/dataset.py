import numpy as np
from torch.utils.data import Dataset
import json
from data.utils import construct_Clinic
import torch

class Clinic(object):
    def __init__(self, cfg):
        super(Clinic, self).__init__()
     
        self.data_path = '/space1/home/lurq/code/clinic/dataset/update.xlsx'
        self.ratio = cfg.TRAINVAL_RATIO
        self.seed = cfg.RANDOM_SEED
    
        np.random.seed(self.seed)
        
        self.trainval = []
        self.train = []
        self.val = []
        self.test = []
        infos = construct_Clinic(self.data_path, cfg.TARGET_NAME, cfg.USE_ICON)
        feats = infos['feat']
        targets = infos['target']

        print('Feature dimension: ', feats.shape[1])
           
        # Split dataset into trainval and test set
        positive_mask = np.where(targets == 1)[0]
        negative_mask = np.where(targets == 0)[0] 
        trainval_mask = []
        test_mask = []
        for pick in (positive_mask, negative_mask):
            total = len(pick)
            rand_uniform = np.random.uniform(0, 1, total)
            trainval_mask.extend(pick[rand_uniform < self.ratio])
            test_mask.extend(pick[rand_uniform >= self.ratio])
        # Further split trainval set into train and val set
        train_mask = []
        val_mask = []
        rand_uniform = np.random.uniform(0, 1, len(trainval_mask))
        train_mask.extend(np.array(trainval_mask)[rand_uniform<0.8])
        val_mask.extend(np.array(trainval_mask)[rand_uniform>=0.8])
          
        print('Trainval mask:', trainval_mask)
        print('test mask:', test_mask)
        print('{:>23}{:>10}{:>7}{:>5}{:>6}'\
              .format('Total', 'Trainval', 'Train', 'Val', 'Test'))
        print('Negative counts: {:>6}{:>10}{:>7}{:>5}{:>6}'\
              .format(np.sum(targets==0), 
                      np.sum(targets[trainval_mask]==0),
                      np.sum(targets[train_mask]==0),
                      np.sum(targets[val_mask]==0),
                      np.sum(targets[test_mask]==0)))
        print('Positive counts: {:>6}{:>10}{:>7}{:>5}{:>6}'\
              .format(np.sum(targets==1), 
                      np.sum(targets[trainval_mask]==1),
                      np.sum(targets[train_mask]==1),
                      np.sum(targets[val_mask]==1),
                      np.sum(targets[test_mask]==1)))
        
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
