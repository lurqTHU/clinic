import torch
from model import BPnet
from data import build_dataloader
from config import cfg
import argparse
import os
from evaluate import Acc
import logging
import sys


def calculate_contribution(config):
    logger = logging.getLogger('clinic')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    device = config.DEVICE
 
    train_loader, val_loader, feat_dim = build_dataloader(config)
    
    model = BPnet(num_layers=config.NUM_LAYERS, in_planes=feat_dim,
                  mid_planes=config.MID_PLANES, activation_type=config.ACTIVATION)

    model.load_param(config.TEST_WEIGHT)
    model.to(device)
    model.eval()

    cumulator = torch.zeros(feat_dim, dtype=torch.float32)
    cumulator = cumulator.to(device)
    
    for iteration, (feat, target) in enumerate(val_loader):
        feat = feat.to(device)
        feat.requires_grad = True
        score = model(feat)
        score = torch.sigmoid(score)
        for i in range(score.shape[0]):
            score[i].backward(retain_graph=True)
        cumulator += torch.sum(torch.pow(feat.grad, 2), dim=0)

    for iteration, (feat, target) in enumerate(train_loader):
        feat = feat.to(device)
        feat.requires_grad = True
        score = model(feat)
        score = torch.sigmoid(score)
        for i in range(score.shape[0]):
            score[i].backward(retain_graph=True)
        cumulator += torch.sum(torch.pow(feat.grad, 2), dim=0)

    logger.info('The contribution of each input element: {}'.format(cumulator))


def main():
    parser = argparse.ArgumentParser(description='Clinic Calculating Contribution')
    parser.add_argument('--cfg_file', default=None, type=str)
    
    args = parser.parse_args()
    
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.freeze()
  
    if cfg.DEVICE == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID 
    
    calculate_contribution(cfg)


if __name__ == '__main__':
    main()
