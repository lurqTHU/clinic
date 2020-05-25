import torch
from model import BPnet
from data import build_dataloader
from config import cfg
import argparse
import os
from evaluate import Acc
import logging
import sys
import sys
sys.path.append('../')
from tools import analyze_roc, calculate_95CI 
import numpy as np


roc_name_dict = {'vas': 'Pain', 'sas': 'Anxiety', 
                 'qol': 'Quality of Life'} 

def test(config, output_dir, test_weight, plot_roc=False):

    device = config.DEVICE
 
    trainval_loader, train_loader, val_loader,\
        test_loader, feat_dim = build_dataloader(config)
    
    all_loaders = (trainval_loader, train_loader, val_loader,
                   test_loader)
    
    model = BPnet(num_layers=config.NUM_LAYERS, in_planes=feat_dim,
                  mid_planes=config.MID_PLANES, 
                  activation_type=config.ACTIVATION)

    model.load_param(test_weight)
    model.to(device)

    evaluator = Acc(thres=config.THRES, metric=config.VAL_METRIC)
    
    model.eval()
 
    all_acc = []
    all_fpr = []
    all_tpr = []
    all_thres = [] 
    # Evaluate trainval, train, val and test subsets
    for data_loader in all_loaders: 
        evaluator.reset()
        for iteration, (feat, target) in enumerate(data_loader):
            with torch.no_grad():
                feat = feat.to(device)
                target = target.to(device)
                score = model(feat)
                evaluator.update((score, target))
        acc, fpr, tpr, thresholds = evaluator.compute()

        all_acc.append(acc)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_thres.append(thresholds)
  
    # Only calculate AUC for trainval set
    auc, optimum = analyze_roc(all_fpr[0], all_tpr[0], all_thres[0], 
                               plot_path=output_dir, 
                               target_name=roc_name_dict[cfg.TARGET_NAME], 
                               plot=plot_roc)
    return auc, optimum, all_acc


def multi_test(cfg, output_dir):
    logger = logging.getLogger('clinic')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    test_folder = cfg.TEST_FOLDER
    test_prefix = cfg.TEST_PREFIX
    model_weights = os.listdir(test_folder)
   
    all_auc = []
    all_optimum = []
    all_train_acc = []
    all_val_acc = []
    all_test_acc = []
    for weight in model_weights:
        if weight.startswith(test_prefix) and weight.endswith('.pth'):
            weight = os.path.join(test_folder, weight)
            auc, optimum, acc = test(cfg, output_dir, weight)
            all_auc.append(auc)
            all_optimum.append(optimum)
            all_train_acc.append(acc[1])
            all_val_acc.append(acc[2])
            all_test_acc.append(acc[3])
   
    auc = calculate_95CI(np.array(all_auc), return_idx=True)
    optimum = calculate_95CI(np.array(all_optimum).transpose(1,0))
    train_acc = calculate_95CI(np.array(all_train_acc))
    val_acc = calculate_95CI(np.array(all_val_acc))
    test_acc = calculate_95CI(np.array(all_test_acc))
   
    # Plot ROC curve of the median AUC
    idx = 0 
    for weight in model_weights:
        if weight.startswith(test_prefix) and weight.endswith('.pth'):
            if idx == auc[2]:
                weight = os.path.join(test_folder, weight)
                test(cfg, output_dir, weight, plot_roc=True)
                break
            else:
                idx += 1
 
    logger.info('Median of AUC: {:.3f}, 95% CI: [{:.3f}, {:.3f}]'\
                .format(auc[0][0], auc[1][0,0], auc[1][0,1]))
    logger.info('Median of Sensitivity: {:.3f}, 95% CI: [{:.3f}, {:.3f}]'\
                .format(optimum[0][0], optimum[1][0,0], optimum[1][0,1]))
    logger.info('Median of Specificity: {:.3f}, 95% CI: [{:.3f}, {:.3f}]'\
                .format(optimum[0][1], optimum[1][1,0], optimum[1][1,1]))
    logger.info('Median of train acc: {:.3f}, 95% CI: [{:.3f}, {:.3f}]'\
                .format(train_acc[0][0], train_acc[1][0,0], train_acc[1][0,1]))
    logger.info('Median of val acc: {:.3f}, 95% CI: [{:.3f}, {:.3f}]'\
                .format(val_acc[0][0], val_acc[1][0,0], val_acc[1][0,1]))
    logger.info('Median of test acc: {:.3f}, 95% CI: [{:.3f}, {:.3f}]'\
                .format(test_acc[0][0], test_acc[1][0,0], test_acc[1][0,1]))
           

def main():
    parser = argparse.ArgumentParser(description='Clinic Testing')
    parser.add_argument('--cfg_file', default=None, type=str)
    
    args = parser.parse_args()
    
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.freeze()

    if cfg.DEVICE == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID

    experiment_name = 'no_config'
    if args.cfg_file != "":
        experiment_name = args.cfg_file.split('/')[-1].split('.yml')[0]

    output_dir = './output/' + experiment_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    multi_test(cfg, output_dir)


if __name__ == '__main__':
    main()
