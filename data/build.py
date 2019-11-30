import torch
from torch.utils.data import DataLoader
from dataset import Clinic, ImageDataset 


def train_collate_fn(batch):
    feats, targets = zip(*batch)
    return torch.stack(feats, dim=0), targets


def build_dataloader(cfg):
    train_batch = cfg.TRAIN_BATCH_SIZE
    val_batch = cfg.VAL_BATCH_SIZE
    
    dataset = Clinic(cfg)
    
    train_set = ImageDataset(dataset.train)
    train_loader = DataLoader(train_set, 
                              batch_size=train_batch,
                              shuffle=True,
                              collate_fn=train_collate_fn)

    val_set = ImageDataset(dataset.val)
    val_loader = DataLoader(val_set, 
                            batch_size=train_batch,
                            shuffle=False,
                            collate_fn=train_collate_fn)

    return train_loader, val_loader

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from config import cfg
    train_loader, val_loader = build_dataloader(cfg)
    import pdb;pdb.set_trace()