import os
import torch.utils.data as data
from torchvision.io import read_image
import config

class Celeba(data.Dataset):
    """
    """
    def __init__(self, sys_config=config.SysConfig()):
        super(Celeba, self).__init__()
        
        self.files = []
        
        count = 0
        for dir_path, _, files in os.walk(sys_config.path_celeba_root):
            if count == 2:
                break
            
            for file in files:
                if count == 2:
                        break
                    
                if '.jpg' in file:
                    
                    self.files.append(
                        os.path.join(dir_path, file)
                    )
                    count = count + 1
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        path = self.files[idx]
        img = read_image(path)
        img = img / 255.
        return img