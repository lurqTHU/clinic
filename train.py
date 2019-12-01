import torch
from model import BPnet
from solver import build_optimizer, MultiStepLr 
from data import build_dataloader
from config import cfg
from loss import build_loss
import argparse
import os
from evaluate import Acc

def train(config):
    train_loader, val_loader, feat_dim = build_dataloader(config)

    model = BPnet(num_layers=config.NUM_LAYERS, in_planes=feat_dim, mid_planes=config.MID_PLANES)
    
    loss_fn = build_loss(config)
    optimizer = build_optimizer(config, model)
    scheduler = MultiStepLr(optimizer, config.STEPS,
                            config.GAMMA)

    epochs = config.MAX_EPOCHS
    log_period = config.LOG_PERIOD
    eval_period = config.EVAL_PERIOD
    device = config.DEVICE

    evaluator = Acc(thres=config.THRES, metric=config.VAL_METRIC)
 
    model.to(device)   

    for epoch in range(1, epochs+1):
        scheduler.step()

        print('Epoch: ', epoch, 'Learning Rate: {:.2e}'.format(scheduler.get_lr()[0]))

        model.train()
        for iteration, (feat, target) in enumerate(train_loader):
            optimizer.zero_grad()
            feat = feat.to(device)
            target = target.to(device)
            score = model(feat)
            loss = loss_fn(score, target)
         
            loss.backward()
            optimizer.step()
            
        if epoch % eval_period == 0:
            model.eval()
            for iteration, (feat, target) in enumerate(val_loader):
                with torch.no_grad():
                    feat = feat.to(device)
                    score = model(feat)
                    evaluator.update((score, target))
            evaluator.compute()
            

def main():
    parser = argparse.ArgumentParser(description='Clinic Training')
    parser.add_argument('--cfg_file', default=None, type=str)
    
    args = parser.parse_args()

    if args.cfg_file:    
        cfg.merge_from_file(args.cfg_file)
    cfg.freeze()

    if cfg.DEVICE == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID
    
    train(cfg)
   


if __name__ == '__main__':
    main()
