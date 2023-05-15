import torch
import torch.nn as nn

class DatasetNormalization(nn.Module):
    """
    Just add this layer as the first layer of your network
    and it will normalize your raw features.
    But do not forget to take log of counters-like features by yourself.
    """
    def init(self, x, nonorm_suffix_size=0):
        super().init()
        mu = torch.mean(x, dim=(-2,))
        std = torch.clamp(torch.std(x, dim=(-2,)), min=0.001)
        if nonorm_suffix_size > 0:
            mu = torch.cat((mu, torch.zeros(nonorm_suffix_size)))
            std = torch.cat((std, torch.ones(nonorm_suffix_size)))
        self.p_mu = nn.Parameter(mu, requires_grad=False)
        self.p_std = nn.Parameter(std, requires_grad=False)
    
    def forward(self, x):
        return (x - self.p_mu) / self.p_std