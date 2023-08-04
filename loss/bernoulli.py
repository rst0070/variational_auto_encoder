import torch.nn as nn
import torch

class BernoulliLoss(nn.Module):
    
    def __init__(self):
        super(BernoulliLoss, self).__init__()
        
    def forward(self, x, p, mean, std):
        """_summary_

        Args:
            x (_type_): input data
            p (_type_): output of bernoulli decoder
            mean (_type_): mean tensor from gaussian encoder
            std (_type_): std tensor from gaussian encoder
        """
        x = torch.flatten(input=x, start_dim=1)
        p = torch.flatten(input=p, start_dim=1)
        #print(x.shape, p.shape)
        
        reconstruction_err = - (1 - x) * torch.log10(1 - p + 1e-8) - x * torch.log10(p + 1e-8)
        reconstruction_err = torch.sum(reconstruction_err)
        
        regularization_err = 0.5 * (mean**2 + std**2 - torch.log(std**2 + 1e-8) - 1)
        regularization_err = torch.sum(regularization_err)
        
        return reconstruction_err + regularization_err