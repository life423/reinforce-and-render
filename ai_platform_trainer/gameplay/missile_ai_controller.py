import math
import torch
from typing import List, Dict, Optional

from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.ai_model.simple_missile_model import SimpleMissileModel

def update_missile_ai(
    missiles: List["Missile"],
    player_pos: Dict[str, float],
    enemy_pos: Optional[Dict[str, float]],
    shared_input_tensor: torch.Tensor,
    missile_model: SimpleMissileModel
) -> None:
    if not enemy_pos:
        return

    for missile in missiles:
        current_angle = math.atan2(missile.vy, missile.vx)

        px, py = player_pos["x"], player_pos["y"]
        ex, ey = enemy_pos["x"], enemy_pos["y"]
        dist_val = math.hypot(px - ex, py - ey)

        # Prepare model input
        shared_input_tensor[0, 0] = px
        shared_input_tensor[0, 1] = py
        shared_input_tensor[0, 2] = ex
        shared_input_tensor[0, 3] = ey
        shared_input_tensor[0, 4] = missile.pos["x"]
        shared_input_tensor[0, 5] = missile.pos["y"]
        shared_input_tensor[0, 6] = current_angle
        shared_input_tensor[0, 7] = dist_val
        shared_input_tensor[0, 8] = 0.0

        with torch.no_grad():
            angle_delta = missile_model(shared_input_tensor).item()

        # Print the raw model output for debugging
        print("angle_delta from model =", angle_delta)

        # Apply the angle delta
        desired_angle = math.atan2(enemy_pos["y"] - missile.pos["y"], enemy_pos["x"] - missile.pos["x"])
        # Compute the difference between the desired angle and the current angle.
        angle_diff = desired_angle - current_angle
        # Blend the model's output with the actual target difference (using a factor for tuning).
        blended_delta = 0.5 * angle_diff + 0.5 * angle_delta
        max_delta = math.radians(5)  # maximum 5° per frame
        constrained_delta = max(-max_delta, min(max_delta, blended_delta))
        new_angle = current_angle + constrained_delta
        speed = math.hypot(missile.vx, missile.vy)
        missile.vx = math.cos(new_angle) * speed
        missile.vy = math.sin(new_angle) * speed

        # Also store angle_delta on the missile object if you want to log it later
        missile.last_action = angle_delta