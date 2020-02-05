import torch
from model import BPnet
from solver import build_optimizer, MultiStepLr 
from data import build_dataloader
from config import cfg
from loss import build_loss
import argparse
import os
from evaluate import Acc, AverageMeter
import logging
from utils.logger import setup_logger
from utils.plot_curves import plot_curve

def train(config, output_dir):
    logger = logging.getLogger('clinic.train')
 
    train_loader, val_loader, feat_dim = build_dataloader(config)

    model = BPnet(num_layers=config.NUM_LAYERS, in_planes=feat_dim, 
                  mid_planes=config.MID_PLANES, activation_type=config.ACTIVATION)
    
    loss_fn = build_loss(config)
    optimizer = build_optimizer(config, model)
    scheduler = MultiStepLr(optimizer, config.STEPS,
                            config.GAMMA)

    epochs = config.MAX_EPOCHS
    log_period = config.LOG_PERIOD
    eval_period = config.EVAL_PERIOD
    save_period = config.CHECKPOINT_PERIOD
    device = config.DEVICE

    evaluator = Acc(thres=config.THRES, metric=config.VAL_METRIC)
    loss_meter = AverageMeter() 

    model.to(device)   

    for epoch in range(1, epochs+1):
        scheduler.step()
        
        evaluator.reset()
        loss_meter.reset()
        
        model.train()
        for iteration, (feat, target) in enumerate(train_loader):
            optimizer.zero_grad()
            feat = feat.to(device)
            target = target.to(device)
            score = model(feat)
            loss = loss_fn(score, target)
         
            loss.backward()
            optimizer.step()
 
            loss_meter.update(loss.item(), feat.shape[0])

            if (iteration + 1) % log_period == 0:
                logger.info('Epoch[{}/{}] Iteration[{}/{}] Loss: {:.3f}, Lr: {:.2e}'
                    .format(epoch, epochs, (iteration+1), len(train_loader), 
                            loss_meter.avg, scheduler.get_lr()[0]))
            
        if epoch % eval_period == 0:
            model.eval()
            for iteration, (feat, target) in enumerate(val_loader):
                with torch.no_grad():
                    feat = feat.to(device)
                    score = model(feat)
                    evaluator.update((score, target))
            evaluator.compute()
        
        if epoch % save_period == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_{}.pth'.format(epoch))) 


def main():
    parser = argparse.ArgumentParser(description='Clinic Training')
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

    logger, log_path = setup_logger('clinic', output_dir,
                                    experiment_name)
    logger.info('Training with config:\n{}'.format(cfg))

    if cfg.DEVICE == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID
    
    train(cfg, output_dir)

    plot_curve(log_path, experiment_name, output_dir)
   

if __name__ == '__main__':
    main()
