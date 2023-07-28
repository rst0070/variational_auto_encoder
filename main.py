import wandb
import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
import config
import os
from trainer import Trainer
from model.mlp import BernoulliMLP

class Main:
    
    def __init__(self):
        
        sys_config = config.SysConfig()
        exp_config = config.ExpConfig()
        
        self.path_save = sys_config.path_save
        
        ###
        ###    seed setting
        ###
        seed = sys_config.random_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        #  cudnn setting to enhance speed of dilated conv
        cudnn.deterministic = False
        cudnn.benchmark = True
        
        # cudnn.deterministic = True
        # cudnn.benchmark = False
        
        
        ###
        ###    wandb setting
        ###
        if sys_config.wandb_disabled:
            os.system("wandb disabled")
            
        os.system(f"wandb login {sys_config.wandb_key}")
        wandb.init(
            project = sys_config.wandb_project,
            entity  = sys_config.wandb_entity,
            name    = sys_config.wandb_name
        )
        
        
        ###
        ###    training environment setting
        ###
        self.max_epoch = exp_config.max_epoch
        self.model = BernoulliMLP(sys_config.device).to(sys_config.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = exp_config.lr,
            weight_decay = exp_config.weight_decay,
            amsgrad=exp_config.amsgrad
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=exp_config.lr_sch_step_size,
            gamma=exp_config.lr_sch_gamma
        )
        
        self.trainer = Trainer(model=self.model, optimizer=self.optimizer)
        
        
    def start(self):
        
        for epoch in range(1, self.max_epoch + 1):
            
            # --------------- train --------------- #
            self.trainer.train()
            
            self.lr_scheduler.step()  
            
            model_state = self.model.state_dict()
            torch.save(model_state, self.path_save)
            
        

if __name__ == '__main__':
    program = Main()
    program.start()
    