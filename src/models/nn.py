import torch
import torch.nn as nn



class DemandNet1(nn.Module):
    """
    This network is the first to try.
    + DatasetNormalization
    + Some Linear + ReLU, one LayerNorm inside
    + one residual connection
    """
    constants = ['output_size']
    def __init__(
        self, 
        x_norm_prefix, # train_dataset without embeddings,
                       # for DatasetNormlization layer initialization
        mid_size: int = 20,
        embed_size: int = 0,
        output_size: int = 1,
        activation: nn.Module = nn.ReLU()
    ):
        super().__init__()
        input_size = len(x_norm_prefix[0]) + embed_size
        self.norm = DatasetNormalization(x_norm_prefix, embed_size)
        self.output_size = output_size
        self.leg = nn.Sequential(
            nn.Linear(input_size, mid_size),
            activation,
            nn.Linear(mid_size, mid_size),
            nn.LayerNorm(mid_size),
            activation,
            nn.Linear(mid_size, mid_size),
            activation,
            nn.LayerNorm(mid_size)
        )
        self.body = nn.Sequential(
            nn.Linear(input_size + mid_size, mid_size),
            activation,
            nn.Linear(mid_size, mid_size),
            activation,
            nn.Linear(mid_size, output_size)
        )
    
    def forward(self, x):
        x1 = self.norm(x)
        x2 = self.leg(x1)
        x = torch.cat((x1, 4 * x2), dim=1)
        yp = self.body(x)
        return torch.squeeze(yp) if self.output_size == 1 else yp