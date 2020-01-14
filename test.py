import torch
from model import BPnet
from data import build_dataloader
from config import cfg
import argparse
import os
from evaluate import Acc
import logging
import sys


def test(config):
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

    evaluator = Acc(thres=config.THRES, metric=config.VAL_METRIC)
    
    for iteration, (feat, target) in enumerate(val_loader):
        with torch.no_grad():
            feat = feat.to(device)
            target = target.to(device)
            score = model(feat)
            evaluator.update((score, target))
    acc = evaluator.compute()


def main():
    parser = argparse.ArgumentParser(description='Clinic Testing')
    parser.add_argument('--cfg_file', default=None, type=str)
    
    args = parser.parse_args()
    
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.freeze()
    
    test(cfg)


if __name__ == '__main__':
    main()
