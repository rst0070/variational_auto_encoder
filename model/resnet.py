import torch
import torch.nn as nn

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1): #projection shortcut
        super(Resblock, self).__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        x = self.relu(self.residual_function(x) + self.shortcut(x))
        return x

class TResblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1): #projection shortcut
        super(TResblock, self).__init__()
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        if stride != 1:
            self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 2), stride=stride, padding=0, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        x = self.residual_function(x)
        
        return self.conv(x + identity)
        

  
class ResNet34Bernoulli(nn.Module):
    
    def __init__(self, device):
        
        super(ResNet34Bernoulli, self).__init__()
        
        self.device = device
        self.J = 256
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
            
            Resblock(64, 64),
            Resblock(64, 64),
            Resblock(64, 64),
        
            Resblock(64, 128, stride=2),
            Resblock(128, 128),
            Resblock(128, 128),
            Resblock(128, 128),
            
            Resblock(128, 256, stride=2),
            Resblock(256, 256),
            Resblock(256, 256),
            Resblock(256, 256),
            Resblock(256, 256),
            Resblock(256, 256),
            
            Resblock(256, 512, stride=2),
            Resblock(512, 512),
            Resblock(512, 512)
        )
        
        self.decoder = nn.Sequential(
            TResblock(256, 256),
            TResblock(256, 256),
            TResblock(256, 256, stride=2),
            
            TResblock(256, 256),
            TResblock(256, 256),
            TResblock(256, 128, stride=2),
            
            TResblock(128, 128),
            TResblock(128, 128),
            TResblock(128, 128),
            TResblock(128, 64, stride=2),
            
            TResblock(64, 64),
            TResblock(64, 64),
            TResblock(64, 64),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0),
        )
        
    def forward(self, x):
        
        code = self.encoder(x)
        mean, std = code[:, 0 : self.J, :, :], code[:, self.J : , :, :]
        epsilon = torch.normal(mean=torch.zeros_like(mean), std=torch.ones_like(std)).to(self.device)
        
        z = mean + std * epsilon
        p = self.decoder(z)
        p = torch.sigmoid(p)
        
        return p, mean, std
    
if __name__ == "__main__":
    from torchinfo import summary
    
    model = ResNet34Bernoulli('cpu').to('cpu')
    summary(model, (2, 3, 128, 128), device='cpu')