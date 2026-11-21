import torch
import torch.nn as nn

class GrayWorldLoss(nn.Module):
    def __init__(self):
        super(GrayWorldLoss, self).__init__()

    def forward(self, x):
        # Channel-wise mean calculation (R, G, B)
        mu_r = torch.mean(x[:, 0, :, :], dim=[1, 2])
        mu_g = torch.mean(x[:, 1, :, :], dim=[1, 2])
        mu_b = torch.mean(x[:, 2, :, :], dim=[1, 2])
        
        mu_global = (mu_r + mu_g + mu_b) / 3.0
        
        # Mean Squared Difference across channel means and global mean
        loss = (mu_r - mu_global)**2 + (mu_g - mu_global)**2 + (mu_b - mu_global)**2
        return torch.mean(loss)