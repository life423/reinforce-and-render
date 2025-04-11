import math
import random
from typing import Dict, List, Optional, Tuple

from ai_platform_trainer.core.config_manager import get_config_manager


# Get the ConfigManager instance
config_manager = get_config_manager()


def calculate_spawn_position(
    screen_width: int,
    screen_height: int,
    entity_size: int,
    spawn_area: str = "edge",
    margin: int = 50
) -> Dict[str, int]:
    """
    Calculate a position to spawn an entity based on the specified area.
    
    Args:
        screen_width: Width of the screen
        screen_height: Height of the screen
        entity_size: Size of the entity to spawn
        spawn_area: Area to spawn in ('edge', 'top', 'random', etc.)
        margin: Margin from the edge of the screen
        
    Returns:
        Dictionary with x and y coordinates
    """
    if spawn_area == "edge":
        # Randomly choose one of the four edges
        edge = random.choice(["top", "bottom", "left", "right"])
    else:
        edge = spawn_area
        
    x, y = 0, 0
    
    if edge == "top":
        x = random.randint(margin, screen_width - entity_size - margin)
        y = margin
    elif edge == "bottom":
        x = random.randint(margin, screen_width - entity_size - margin)
        y = screen_height - entity_size - margin
    elif edge == "left":
        x = margin
        y = random.randint(margin, screen_height - entity_size - margin)
    elif edge == "right":
        x = screen_width - entity_size - margin
        y = random.randint(margin, screen_height - entity_size - margin)
    elif edge == "random":
        x = random.randint(margin, screen_width - entity_size - margin)
        y = random.randint(margin, screen_height - entity_size - margin)
    
    return {"x": x, "y": y}


def create_enemy_spawn_positions(
    count: int,
    screen_width: int,
    screen_height: int,
    enemy_size: int,
    min_distance: int = 100
) -> List[Dict[str, int]]:
    """
    Create multiple enemy spawn positions ensuring minimum distance between them.
    
    Args:
        count: Number of enemies to spawn
        screen_width: Width of the screen
        screen_height: Height of the screen
        enemy_size: Size of the enemy
        min_distance: Minimum distance between enemies
        
    Returns:
        List of dictionaries with x and y coordinates
    """
    positions = []
    attempts = 0
    max_attempts = 100
    
    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        new_pos = calculate_spawn_position(
            screen_width=screen_width,
            screen_height=screen_height,
            entity_size=enemy_size,
            spawn_area="random"
        )
        
        # Check distance from all existing positions
        valid = True
        for pos in positions:
            dist = math.hypot(new_pos["x"] - pos["x"], new_pos["y"] - pos["y"])
            if dist < min_distance:
                valid = False
                break
                
        if valid:
            positions.append(new_pos)
            
    return positions


def find_valid_spawn_position(
    screen_width: int,
    screen_height: int,
    entity_size: int,
    margin: int = None,
    min_dist: int = None,
    other_pos: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """
    Returns a random (x, y) within the screen boundaries, ensuring optional
    minimum distance from another position.
    """
    # Use the config values if not provided
    if margin is None:
        margin = config_manager.get("gameplay.wall_margin", 50)
    if min_dist is None:
        min_dist = config_manager.get("gameplay.min_distance", 100)
    x_min = margin
    x_max = screen_width - entity_size - margin
    y_min = margin
    y_max = screen_height - entity_size - margin

    while True:
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)

        if other_pos:
            dist = math.hypot(x - other_pos[0], y - other_pos[1])
            if dist >= min_dist:
                return (x, y)
        else:
            return (x, y)


def find_enemy_spawn_position(
    screen_width: int,
    screen_height: int,
    enemy_size: int,
    player_pos: Tuple[float, float]
) -> Tuple[int, int]:
    """
    Specialized helper for spawning an enemy, ensuring min distance from the player.
    """
    wall_margin = config_manager.get("gameplay.wall_margin", 50)
    min_distance = config_manager.get("gameplay.min_distance", 100)

    return find_valid_spawn_position(
        screen_width=screen_width,
        screen_height=screen_height,
        entity_size=enemy_size,
        margin=wall_margin,
        min_dist=min_distance,
        other_pos=player_pos,
    )
