import torch

class ExpConfig:
    """
    
    """
    def __init__(self):
        
        self.max_epoch              =  40
        self.batch_size             =  100
        # ------------------ optimizer setting ------------------ #
        self.lr                     =   1e-3
        self.lr_min                 =   1e-7
        self.weight_decay           =   1e-4
        self.amsgrad                =   True
        # ------------------ learning rate scheduler ------------------ #
        self.lr_sch_step_size       =   1       
        self.lr_sch_gamma           =   0.94
        
class SysConfig:
    
    def __init__(self):
        # ------------------ path of voxceleb1 ------------------ #
        self.path_celeba_root       = '/data/celeba'        
        self.path_save              = '/result/BernoulliMLP.pt'
        # ------------------ wandb setting ------------------ #
        self.wandb_disabled         = True
        self.wandb_key              = '8c8d77ae7f92de2b007ad093af722aaae5f31003'
        self.wandb_project          = 'data_aug'
        self.wandb_entity           = 'rst0070'
        self.wandb_name             = ''
        # ------------------ device setting ------------------ #
        self.num_workers            = 4
        self.device                 =   'cuda:0'
        """device to use for training and testing"""
        
        self.random_seed            = 1234

if __name__ == "__main__":
    exp = ExpConfig()
    print(exp.n_mels)