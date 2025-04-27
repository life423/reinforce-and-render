"""
Test script for VecRLBounceEnv class.
"""
import torch
from ai_platform_trainer.agents.vec_env import VecRLBounceEnv


def test_vec_env():
    print("Testing VecRLBounceEnv...")
    
    # Test parameters
    num_envs = 4  # Use a small number for testing
    device = "cpu"
    
    # Create the vectorized environment
    vec_env = VecRLBounceEnv(num_envs=num_envs, device=device)
    
    # Reset and check observation shape
    obs = vec_env.reset()
    print(
        f"Observation shape: {obs.shape}, "
        f"dtype: {obs.dtype}, device: {obs.device}"
    )
    assert obs.shape == (num_envs, 6)
    assert obs.dtype == torch.float32
    assert str(obs.device) == device
    
    # Take a random action and check returns
    actions = torch.rand(num_envs, 2) * 2 - 1  # Random actions in [-1, 1]
    next_obs, rewards, dones = vec_env.step(actions)
    
    print(f"Next obs shape: {next_obs.shape}")
    print(f"Rewards shape: {rewards.shape}, values: {rewards}")
    print(f"Dones shape: {dones.shape}, values: {dones}")
    
    assert next_obs.shape == (num_envs, 6)
    assert rewards.shape == (num_envs,)
    assert dones.shape == (num_envs,)
    
    print("All tests passed!")


if __name__ == "__main__":
    test_vec_env()
