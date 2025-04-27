import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, inp: int, out: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out),
        )
    
    def forward(self, x): 
        return self.net(x)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int = 6, act_dim: int = 2):
        super().__init__()
        self.feat = MLP(obs_dim, 64)
        self.mean = nn.Linear(64, act_dim)
        # learned log-std parameter (one per action dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        h = torch.tanh(self.feat(obs))
        mean = self.mean(h)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int = 6):
        super().__init__()
        self.net = MLP(obs_dim, 1)
    
    def forward(self, obs): 
        return self.net(obs).squeeze(-1)
