"""
Enemy AI Controller for AI Platform Trainer.

This module handles the enemy's AI-driven movement using either
the trained neural network model or reinforcement learning policy.
"""
import math
import random
import logging
import time
import torch
import os
from typing import Tuple, Dict, List


class EnemyAIController:
    """
    Controller for enemy AI behavior.

    This class manages the enemy movement logic, providing different strategies
    including neural network inference, reinforcement learning policy, and
    fallback behaviors to prevent freezing.
    """

    def __init__(self):
        """Initialize the enemy AI controller."""
        self.last_action_time = time.time()
        self.action_interval = 0.05  # 50ms between actions
        self.smoothing_factor = 0.7  # For smoothing movements
        self.prev_dx = 0
        self.prev_dy = 0
        self.stuck_counter = 0
        self.prev_positions: List[Dict[str, float]] = []
        self.max_positions = 10  # Store last 10 positions to detect being stuck

        # RL model path
        self.rl_model_path = "models/enemy_rl/final_model.zip"
        self._rl_model = None  # Lazy-loaded

    @property
    def rl_model(self):
        """Lazy-load the RL model if available."""
        if self._rl_model is None and os.path.exists(self.rl_model_path):
            try:
                # Only import these when needed to avoid circular imports
                from stable_baselines3 import PPO
                self._rl_model = PPO.load(self.rl_model_path)
                logging.info(f"Successfully loaded RL model from {self.rl_model_path}")
            except Exception as e:
                logging.error(f"Failed to load RL model: {e}")
        return self._rl_model

    def update_enemy_movement(
        self,
        enemy,
        player_x: float,
        player_y: float,
        player_speed: float,
        current_time: int
    ) -> None:
        """
        Handle the enemy's AI-driven movement.

        Args:
            enemy: EnemyPlay instance
            player_x: Player's x position
            player_y: Player's y position
            player_speed: Player's movement speed
            current_time: Current game time
        """
        # If the enemy is not visible, skip
        if not enemy.visible:
            return

        # Throttle updates for better performance
        current_time_sec = time.time()
        if current_time_sec - self.last_action_time < self.action_interval:
            return
        self.last_action_time = current_time_sec

        # Track position history to detect if enemy is stuck
        self._update_position_history(enemy.pos)

        # Try using reinforcement learning model if available
        if self.rl_model is not None:
            action_dx, action_dy = self._get_rl_action(enemy, player_x, player_y)
        else:
            # Fall back to neural network if no RL model
            action_dx, action_dy = self._get_nn_action(enemy, player_x, player_y)

        # Check if enemy is stuck and apply special behavior if needed
        if self._is_enemy_stuck():
            action_dx, action_dy = self._handle_stuck_enemy(player_x, player_y, enemy.pos)

        # Apply smoothing
        action_dx = self.smoothing_factor * action_dx + (1 - self.smoothing_factor) * self.prev_dx
        action_dy = self.smoothing_factor * action_dy + (1 - self.smoothing_factor) * self.prev_dy

        # Normalize direction vector
        action_dx, action_dy = self._normalize_vector(action_dx, action_dy)

        # Store for next frame's smoothing
        self.prev_dx, self.prev_dy = action_dx, action_dy

        # Move enemy at 70% of the player's speed
        speed = player_speed * 0.7
        enemy.pos["x"] += action_dx * speed
        enemy.pos["y"] += action_dy * speed

        # Wrap-around logic
        enemy.pos["x"], enemy.pos["y"] = enemy.wrap_position(enemy.pos["x"], enemy.pos["y"])

    def _get_nn_action(
        self,
        enemy,
        player_x: float,
        player_y: float
    ) -> Tuple[float, float]:
        """
        Get action from neural network model.

        Args:
            enemy: Enemy instance
            player_x: Player's x position
            player_y: Player's y position

        Returns:
            Tuple of (dx, dy) movement direction
        """
        # Construct input state for the model
        dist = math.sqrt((player_x - enemy.pos["x"])**2 + (player_y - enemy.pos["y"])**2)
        state = torch.tensor(
            [[player_x, player_y, enemy.pos["x"], enemy.pos["y"], dist]],
            dtype=torch.float32
        )

        # Inference
        try:
            with torch.no_grad():
                action = enemy.model(state)  # shape: [1, 2]

            action_dx, action_dy = action[0].tolist()

            # If action is very small, treat as zero
            if abs(action_dx) < 1e-6 and abs(action_dy) < 1e-6:
                return self._get_random_direction()

            return action_dx, action_dy

        except Exception as e:
            logging.error(f"Neural network inference error: {e}")
            return self._get_random_direction()

    def _get_rl_action(
        self,
        enemy,
        player_x: float,
        player_y: float
    ) -> Tuple[float, float]:
        """
        Get action from reinforcement learning model.

        Args:
            enemy: Enemy instance
            player_x: Player's x position
            player_y: Player's y position

        Returns:
            Tuple of (dx, dy) movement direction
        """
        try:
            # Normalize positions to [0,1] range
            screen_width, screen_height = 800, 600  # Default game screen size

            # Create observation
            obs = [
                enemy.pos["x"] / screen_width,
                enemy.pos["y"] / screen_height,
                player_x / screen_width,
                player_y / screen_height,
            ]

            # Get action from RL model
            action, _ = self.rl_model.predict(obs, deterministic=False)

            # Convert continuous action space to direction
            action_dx, action_dy = action[0], action[1]

            # Ensure the action isn't zero
            if abs(action_dx) < 1e-6 and abs(action_dy) < 1e-6:
                return self._get_random_direction()

            return action_dx, action_dy

        except Exception as e:
            logging.error(f"RL model inference error: {e}")
            return self._get_random_direction()

    def _normalize_vector(self, dx: float, dy: float) -> Tuple[float, float]:
        """
        Normalize a direction vector to unit length.

        Args:
            dx: X component of direction
            dy: Y component of direction

        Returns:
            Normalized (dx, dy) tuple
        """
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            return dx / length, dy / length
        else:
            # Apply a random direction if vector is zero length
            return self._get_random_direction()

    def _get_random_direction(self) -> Tuple[float, float]:
        """
        Get a random unit direction vector.

        Returns:
            Random (dx, dy) direction
        """
        angle = random.uniform(0, 2 * math.pi)
        return math.cos(angle), math.sin(angle)

    def _update_position_history(self, position: Dict[str, float]) -> None:
        """
        Update the history of enemy positions to detect if stuck.

        Args:
            position: Current enemy position
        """
        # Add current position to history
        self.prev_positions.append({"x": position["x"], "y": position["y"]})

        # Limit history size
        if len(self.prev_positions) > self.max_positions:
            self.prev_positions.pop(0)

    def _is_enemy_stuck(self) -> bool:
        """
        Check if the enemy appears to be stuck based on position history.

        Returns:
            True if enemy seems stuck, False otherwise
        """
        if len(self.prev_positions) < self.max_positions:
            return False

        # Calculate variance in positions
        x_positions = [pos["x"] for pos in self.prev_positions]
        y_positions = [pos["y"] for pos in self.prev_positions]

        x_var = max(x_positions) - min(x_positions)
        y_var = max(y_positions) - min(y_positions)

        # If the enemy hasn't moved much, it might be stuck
        if x_var < 10 and y_var < 10:
            self.stuck_counter += 1
            if self.stuck_counter > 3:  # Stuck for several frames
                return True
        else:
            self.stuck_counter = 0

        return False

    def _handle_stuck_enemy(
        self,
        player_x: float,
        player_y: float,
        enemy_pos: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Special behavior for when the enemy is detected as stuck.

        Args:
            player_x: Player's x position
            player_y: Player's y position
            enemy_pos: Enemy's current position

        Returns:
            Direction vector to move the enemy
        """
        logging.debug(f"Enemy detected as stuck at {enemy_pos}, applying escape behavior")

        # Option 1: Move away from player (reversed chase)
        dx = enemy_pos["x"] - player_x
        dy = enemy_pos["y"] - player_y

        # Option 2: Sometimes use completely random movement to break patterns
        if random.random() < 0.3:
            return self._get_random_direction()

        # Normalize the escape vector
        return self._normalize_vector(dx, dy)


# Initialize the controller as a singleton
enemy_controller = EnemyAIController()


def update_enemy_movement(
    enemy,
    player_x: float,
    player_y: float,
    player_speed: float,
    current_time: int
) -> None:
    """
    Legacy function for backward compatibility.
    Delegates to the EnemyAIController instance.

    Args:
        enemy: EnemyPlay instance
        player_x: Player's x position
        player_y: Player's y position
        player_speed: Player's movement speed
        current_time: Current game time
    """
    enemy_controller.update_enemy_movement(
        enemy, player_x, player_y, player_speed, current_time
    )
