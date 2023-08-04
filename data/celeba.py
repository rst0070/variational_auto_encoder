import os
import torch.utils.data as data
from torchvision.io import read_image
import torchvision.transforms.functional as F
import random
import config

class Celeba(data.Dataset):
    """
    """
    def __init__(self, sys_config=config.SysConfig()):
        super(Celeba, self).__init__()
        
        self.files = []
        
        for dir_path, _, files in os.walk(sys_config.path_celeba_root):
            
            for file in files:
    
                if '.jpg' in file:
                    
                    self.files.append(
                        os.path.join(dir_path, file)
                    )
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        path = self.files[idx]
        img = read_image(path)
        img = img / 255.
        return img
    
class CelebaCrop(Celeba):
    """
    """
    def __init__(self, sys_config=config.SysConfig()):
        super(CelebaCrop, self).__init__(sys_config)
    
    def __getitem__(self, idx):
        img = super().__getitem__(idx)
        
        h = 128
        w = 128
        
        h_top = random.randint(0, 218 - h)
        w_top = random.randint(0, 178 - w)
        
        img = img[:, h_top : h + h_top, w_top : w_top + w]
        
        return img