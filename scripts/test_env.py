from ai_platform_trainer.agents.env import RLBounceEnv
import numpy as np

env = RLBounceEnv(seed=1)
obs = env.reset()
cum = 0
for _ in range(300):
    # Generate random action and convert numpy array to tuple
    action_array = np.random.uniform(-1, 1, size=2)
    action = (float(action_array[0]), float(action_array[1]))
    
    obs, r, d, _ = env.step(action)
    cum += r
    if d:
        break
print("episode reward:", cum)
