import torch.nn as nn
import torch.nn.functional as F

class SmoothQLoss(nn.L1Loss):
    constants = ['reduction', 'beta', 'qx']
    def __str__(self):
        return f"{self.__class__.__name__}(beta={self.beta}, qx={self.qx})"
    
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = 'mean',
        qx: float = 0.5,
        beta: float = 0.1,
    ):
        super().__init__(size_average, reduce, reduction)
        self.beta = beta
        self.qx = qx
    
    def forward(
      self, 
      predict: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        m = 2.0 * self.qx - 1.0
        shift = self.beta * m
        if self.reduction == 'mean':
            return (
                F.smooth_l1_loss(
                  target - shift, predict, reduction='mean', 
                  beta=self.beta
                ) + m * torch.mean(target - predict - 0.5 * shift)
            )
        elif self.reduction == 'sum':
            return (
                F.smooth_l1_loss(
                    target - shift, predict, reduction='sum', 
                    beta=self.beta
                ) + m * torch.sum(target - predict - 0.5 * shift)
            )