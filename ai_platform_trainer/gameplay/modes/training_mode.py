import math
import logging
import random
import pygame
from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.utils.helpers import wrap_position
from ai_platform_trainer.gameplay.utils import compute_normalized_direction
from ai_platform_trainer.gameplay.spawner import respawn_enemy_with_fade_in


class TrainingMode:
    def __init__(self, game):
        self.game = game
        self.missile_cooldown = 0
        self.missile_turn_rate = math.radians(3)  # Reduced turn rate
        self.missile_lifespan = {}  # missile -> (birth_time, lifespan)

        # NEW: track each missile's entire frame-by-frame data
        # until collision or expiry
        self.missile_sequences = {}  # missile -> [list of data_points]

    def update(self):
        """Handles updates specifically for training mode"""
        current_time = pygame.time.get_ticks()

        if self.game.enemy and self.game.player:
            enemy_x = self.game.enemy.pos["x"]  # Get these values here
            enemy_y = self.game.enemy.pos["y"]
            self.game.player.update(
                enemy_x, enemy_y
            )  # Pass coordinates to player.update()

            if random.random() < getattr(
                self, "missile_fire_prob", 0.02
            ):  # Missile firing logic (using getattr)
                if self.missile_cooldown > 0:
                    self.missile_cooldown -= 1

                if self.missile_cooldown <= 0 and len(self.game.player.missiles) == 0:
                    self.game.player.shoot_missile(
                        enemy_x, enemy_y
                    )  # Pass enemy position!
                    self.missile_cooldown = 120  # Reset cooldown after shooting

                    if (
                        self.game.player.missiles
                    ):  # Only if a missile was actually shot:
                        msl = self.game.player.missiles[0]  # Get the missile

                        # ... (missile lifespan and sequences setup)

        # Update existing missiles, enemy, and handle collisions (existing logic)
        # ... (rest of your update method, from step 3 onwards in the original code).
        # Ensure all references to shoot_missile() now correctly pass enemy coordinates.

        # Simplified Missile Collision Handling
        if self.game.enemy and self.game.player:
            ex = self.game.enemy.pos["x"]
            ey = self.game.enemy.pos["y"]
            enemy_rect = pygame.Rect(ex, ey, self.game.enemy.size, self.game.enemy.size)
            for msl in self.game.player.missiles[
                :
            ]:  # Iterate over copy to avoid index issues
                if msl.get_rect().colliderect(enemy_rect):
                    logging.info("Missile hit the enemy (training mode).")
                    # finalize as success=1
                    self.finalize_missile_sequence(msl, success=True)  # Keep this call
                    self.game.player.missiles.remove(msl)
                    if msl in self.missile_lifespan:
                        del self.missile_lifespan[msl]
                    self.game.enemy.hide()
                    self.game.is_respawning = True
                    self.game.respawn_timer = current_time + self.game.respawn_delay
                    break  # Stop checking after a hit

        # Handle respawn logic  (Existing code)
        # ...

    def finalize_missile_sequence(self, missile, success: bool) -> None:
        """
        Mark all frames in missile's sequence with 'missile_collision'= success
        Then log them to self.game.data_logger, and clear from missile_sequences.
        """
        if missile not in self.missile_sequences:
            return

        outcome_val = success  # True/False => 1.0/0.0 if you prefer
        frames = self.missile_sequences[missile]

        for frame_data in frames:
            frame_data["missile_collision"] = outcome_val

            # now actually log to data_logger
            if self.game.data_logger:
                self.game.data_logger.log(frame_data)

        # remove from dict
        del self.missile_sequences[missile]
        logging.debug(
            f"Finalized missile sequence with success={success}, frames={len(frames)}"
        )
