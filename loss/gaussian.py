import torch
import torch.nn as nn

class GaussianLoss(nn.Module):
    
    def __init__(self):
        super(GaussianLoss, self).__init__()
        
    def forward(self, x, mean_encoder, std_encoder, mean_decoder, std_decoder):
        """_summary_

        Args:
            x (_type_): input data
            mean (_type_): _description_
            std (_type_): _description_
        """
        
        reconstruction_err = 0.5 * torch.log(std_decoder ** 2) + 0.5 * ((x - mean_decoder)**2 / (std_decoder**2 + 1e-8))
        reconstruction_err = torch.sum(reconstruction_err)
        
        regularization_err = 0.5 * (mean_encoder**2 + std_encoder**2 - torch.log(std_encoder**2 + 1e-8) - 1)
        regularization_err = torch.sum(regularization_err)
        
        return reconstruction_err + regularization_err