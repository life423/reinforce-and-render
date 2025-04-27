from typing import Tuple
import torch
from ai_platform_trainer.agents.env import RLBounceEnv


class VecRLBounceEnv:
    """
    Vectorized wrapper over multiple RLBounceEnv instances.
    All observations are stacked into a (N,6) torch Tensor (GPU-ready).
    """

    def __init__(self, num_envs: int = 32, device: str | torch.device = "cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.envs = [RLBounceEnv(seed=i) for i in range(num_envs)]

        # allocate buffers
        self.obs_buf = torch.zeros(
            (num_envs, 6), dtype=torch.float32, device=self.device
        )
        self.rew_buf = torch.zeros(
            (num_envs,), dtype=torch.float32, device=self.device
        )
        self.done_buf = torch.zeros(
            (num_envs,), dtype=torch.bool, device=self.device
        )

    # ------------------------------------------------------------- #
    def reset(self) -> torch.Tensor:
        for i, e in enumerate(self.envs):
            obs = e.reset()
            self.obs_buf[i] = torch.from_numpy(obs)
            self.done_buf[i] = False
            self.rew_buf[i] = 0.0
        return self.obs_buf

    # ------------------------------------------------------------- #
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        actions: Tensor (N,2) on self.device, values in [-1,1]
        Returns (obs, reward, done) tensors on self.device.
        """
        actions_np = actions.cpu().numpy()
        for i, (e, a) in enumerate(zip(self.envs, actions_np)):
            if self.done_buf[i]:
                # skip already-finished envs (leave obs/reward unchanged)
                continue
            obs, r, d, _ = e.step(a)
            self.obs_buf[i] = torch.from_numpy(obs)
            self.rew_buf[i] = r
            self.done_buf[i] = d

            if d:  # auto-reset so batch never has terminal states
                obs = e.reset()
                self.obs_buf[i] = torch.from_numpy(obs)
                self.done_buf[i] = False

        return self.obs_buf, self.rew_buf, self.done_buf
