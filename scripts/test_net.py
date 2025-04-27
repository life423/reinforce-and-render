# scripts/test_net.py
import torch
from ai_platform_trainer.agents.nn_agent import PolicyNet, ValueNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = PolicyNet().to(device)
value = ValueNet().to(device)

obs = torch.randn(32, 6, device=device)
mean, std = policy(obs)
val = value(obs)

print(mean.shape, std.shape, val.shape)  # (32,2) (32,2) (32,)
