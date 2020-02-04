import torch
from model import BPnet
from data import build_dataloader
from config import cfg
import argparse
import os
from evaluate import Acc
import logging
import sys
from utils.roc import analyze_roc 


def test(config, output_dir):
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
    
    model.eval()
   
    for iteration, (feat, target) in enumerate(val_loader):
        with torch.no_grad():
            feat = feat.to(device)
            target = target.to(device)
            score = model(feat)
            evaluator.update((score, target))
    fpr, tpr, thresholds = evaluator.compute(roc_curve=True)
    analyze_roc(fpr, tpr, thresholds, output_dir, img_name='roc_val.png')

    evaluator.reset()
    for iteration, (feat, target) in enumerate(train_loader):
        with torch.no_grad():
            feat = feat.to(device)
            target = target.to(device)
            score = model(feat)
            evaluator.update((score, target))
    fpr, tpr, thresholds = evaluator.compute(roc_curve=True)
    analyze_roc(fpr, tpr, thresholds, output_dir, img_name='roc_train.png')


def main():
    parser = argparse.ArgumentParser(description='Clinic Testing')
    parser.add_argument('--cfg_file', default=None, type=str)
    
    args = parser.parse_args()
    
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.freeze()

    experiment_name = 'no_config'
    if args.cfg_file != "":
        experiment_name = args.cfg_file.split('/')[-1].split('.yml')[0]

    output_dir = './output/' + experiment_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    test(cfg, output_dir)


if __name__ == '__main__':
    main()
