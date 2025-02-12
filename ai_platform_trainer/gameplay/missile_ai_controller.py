# file: ai_platform_trainer/gameplay/missile_ai_controller.py
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
    """
    Applies AI-driven angle updates for each missile in the player's inventory.
    """
    if not enemy_pos:
        return

    for missile in missiles:
        current_angle = math.atan2(missile.vy, missile.vx)

        px, py = player_pos["x"], player_pos["y"]
        ex, ey = enemy_pos["x"], enemy_pos["y"]
        dist_val = math.hypot(px - ex, py - ey)

        # Reuse the input tensor
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

        new_angle = current_angle + angle_delta
        speed = 5.0
        missile.vx = math.cos(new_angle) * speed
        missile.vy = math.sin(new_angle) * speed