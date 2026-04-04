import torch
import torch.nn as nn


class MLPDiffusionModel(nn.Module):
    """
    A simple MLP used as a score model for diffusion on tabular data.
    """

    def __init__(self, input_dim, hidden_dim=128, depth=3, dropout: float = 0.0):
        super().__init__()
        
        self.input_dim = input_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t=None):
        # timestep ignored for simplicity
        return self.net(x)
