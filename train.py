import torch
from model import BPnet
from solver import build_optimizer, build_scheduler
import config as cfg

def train(config):
    train_loader = build_dataloader(loader_type='train')
    val_loader = build_dataloader(loader_type='val')

    model = BPnet()
    
    optimizer = build_optimizer(model)
    
    scheduler = build_scheduler(config, optimizer)

    epochs = config.MAX_EPOCHS
    log_period = config.LOG_PERIOD
    eval_period = config.EVAL_PERIOD
    device = config.DEVICE
 
    model.to(device)   

    for epoch in range(1, epochs+1):
    
        scheduler.step()

        model.train()
        for iteration, feat, target in enumerate(train_loader):
            optimizer.zero_grad()
            feat = feat.to(device)
            score = model(feat)
            loss = loss_fn(score, target)
         
            loss.backward()
            optimizer.step()
          
            
        if epoch % eval_period == 0:
            model.eval()
            for iteration, feat, target in enumerate(val_loader):
                with torch.no_grad():
                    feat = feat.to(device)
                    score = model(feat)
                    evaluator.update((score, target))
            evluator.compute()
            

def main():
    parser = argparse.ArgumentParser(description='Clinic Training')
    parser.add_argument('--cfg_file', default=None, type=str)
    
    args = parser.parse_args()
    
    cfg.merge_from_file(args.cfg_file)
    cfg.freeze()

    if cfg.DEVICE == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID
    
    train(cfg)
   


if __name__ == '__main__':
    main()
