import torch
from torch.utils.data import DataLoader
from data.dataset import Clinic, ImageDataset 


def train_collate_fn(batch):
    feats, targets = zip(*batch)
    return torch.stack(feats, dim=0), torch.tensor(targets, dtype=torch.float32)


def build_dataloader(cfg):
    train_batch = cfg.TRAIN_BATCH_SIZE
    val_batch = cfg.VAL_BATCH_SIZE
    
    dataset = Clinic(cfg)
    
    trainval_set = ImageDataset(dataset.trainval)
    trainval_loader = DataLoader(trainval_set, 
                                 batch_size=train_batch,
                                 shuffle=True,
                                 collate_fn=train_collate_fn)

    train_set = ImageDataset(dataset.train)
    train_loader = DataLoader(train_set, 
                              batch_size=val_batch,
                              shuffle=True,
                              collate_fn=train_collate_fn)

    val_set = ImageDataset(dataset.val)
    val_loader = DataLoader(val_set, 
                            batch_size=val_batch,
                            shuffle=False,
                            collate_fn=train_collate_fn)
   
    test_set = ImageDataset(dataset.test)
    test_loader = DataLoader(test_set, 
                             batch_size=val_batch,
                             shuffle=False,
                             collate_fn=train_collate_fn)
  
    return trainval_loader, train_loader, val_loader,\
           test_loader, len(dataset.train[0][0])


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from config import cfg
    train_loader, val_loader = build_dataloader(cfg)
    import pdb;pdb.set_trace()
