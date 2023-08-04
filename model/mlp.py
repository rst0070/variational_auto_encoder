import torch.nn as nn
import torch

class BernoulliMLP(nn.Module):
    
    def __init__(self, device, img_size:torch.Size = (3, 218, 178)):
        super(BernoulliMLP, self).__init__()
        
        self.device = device
        self.J = 256
        self.D = 116412 #(3, 218, 178) # 116,412
        self.encoder = nn.Sequential(
            nn.Linear(in_features=116412, out_features=1164),
            nn.ReLU(),
            nn.Linear(in_features=1164, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=self.J * 2)
        )
        
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.J, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=1164),
            nn.ReLU(),
            nn.Linear(in_features=1164, out_features=116412)
        )
        
        
    def forward(self, x):
        
        batch_size = len(x)
        x = x.view(batch_size, -1)
        
        code = self.encoder(x) # [batch, J * 2]
        mean, std = code[:, 0 : self.J], code[:, self.J : ]
        
        epsilon = torch.normal(mean=torch.zeros(size=(batch_size, self.J)), std=torch.ones(size=(batch_size, self.J))).to(self.device)
        
        #print(mean.shape, std.shape, epsilon.shape)
        
        z = mean + std * epsilon
        p = self.decoder(z)
        p = torch.sigmoid(p)
        return p, mean, std