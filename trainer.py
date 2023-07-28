import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.data as data
from data.celeba import Celeba
from loss.bernoulli import BernoulliLoss
import config
import wandb

class Trainer:
    """
    """
    def __init__(self, model:torch.nn.Module, optimizer, sys_config=config.SysConfig(), exp_config=config.ExpConfig()):
                
        self.sys_config = sys_config
        self.exp_config = exp_config
        
        
        ###
        ###    Model and optimizer
        ###
        self.model = model.to(sys_config.device)
        self.loss_fn = BernoulliLoss().to(sys_config.device)
        
        self.optimizer = optimizer
        self.optimizer.add_param_group({
            'params' : self.loss_fn.parameters()
        })
        
        ###
        ###    Select Training dataset between TAN and PAS
        ###
        train_dataset = Celeba()
        
        self.train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=exp_config.batch_size,
            shuffle=True,
            num_workers=sys_config.num_workers,
            pin_memory=True
        )
        '''train data loader.'''
        
        
        
    
    def train(self):
        
        self.model.train()
        self.loss_fn.train()
        
        itered = 0
        loss_sum = 0
        
        pbar = tqdm(self.train_loader)
        for x in pbar:
            
            itered += 1
            
            self.optimizer.zero_grad()
            
            x = x.to(self.sys_config.device)
            
            p, mean, std = self.model(x)
            loss = self.loss_fn(x, p, mean, std)
            
            pbar.set_description(f"train: {loss}")
            loss_sum += loss.detach()
            
            loss.backward()
            self.optimizer.step()
            
            if itered == 50:
                wandb.log({'Loss':loss_sum / float(itered)})
                itered = 0
                loss_sum = 0
        wandb.log({'Loss':loss_sum / float(itered)})