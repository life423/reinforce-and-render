import math
import logging
import random
import pygame
from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.utils.helpers import wrap_position
from ai_platform_trainer.gameplay.utils import compute_normalized_direction


class TrainingMode:
    def __init__(self, game):
        self.game = game
        self.missile_cooldown = 0
        self.missile_lifespan = {}  # missile -> (birth_time, lifespan)
        self.missile_sequences = {}  # missile -> [list of data_points]

    def update(self):
        """Handles updates specifically for training mode."""
        current_time = pygame.time.get_ticks()

        if self.game.enemy and self.game.player:
            enemy_x = self.game.enemy.pos["x"]
            enemy_y = self.game.enemy.pos["y"]

            self.game.player.update(
                enemy_x, enemy_y
            )  # Always update player. Passing coordinates so that it can record training data.
            self.game.enemy.update_movement(
                self.game.player.position["x"],
                self.game.player.position["y"],
                self.game.player.step,
            )  # Always update enemy

            if random.random() < getattr(self, "missile_fire_prob", 0.02):
                if self.missile_cooldown > 0:
                    self.missile_cooldown -= 1

                if self.missile_cooldown <= 0 and len(self.game.player.missiles) == 0:
                    self.game.player.shoot_missile(enemy_x, enemy_y)
                    self.missile_cooldown = 120

                    if (
                        self.game.player.missiles
                    ):  # Only if a missile was actually shot:
                        msl = self.game.player.missiles[0]
                        self.missile_lifespan[msl] = (msl.birth_time, msl.lifespan)
                        self.missile_sequences[msl] = []  # Initialize missile_sequence

        for missile in self.game.player.missiles[
            :
        ]:  # Iterate over a copy for safe removal
            if missile in self.missile_lifespan:
                birth_time, lifespan = self.missile_lifespan[missile]
                if current_time - birth_time >= lifespan:
                    self.finalize_missile_sequence(missile, success=False)
                    self.game.player.missiles.remove(missile)
                    del self.missile_lifespan[missile]
                    logging.debug("Training mode: Missile removed (lifespan expiry).")
                else:  # Missile still active - update pos and check for collision/off-screen
                    missile.update()

                    # Simplified Missile Collision Handling
                    if (
                        self.game.enemy
                    ):  # Ensure enemy exists before accessing properties
                        enemy_rect = pygame.Rect(
                            self.game.enemy.pos["x"],
                            self.game.enemy.pos["y"],
                            self.game.enemy.size,
                            self.game.enemy.size,
                        )

                        if msl.get_rect().colliderect(enemy_rect):
                            logging.info("Missile hit the enemy (training mode).")
                            self.finalize_missile_sequence(msl, success=True)
                            self.game.player.missiles.remove(msl)

                            del self.missile_lifespan[
                                msl
                            ]  # Remove from lifespan tracking
                            self.game.respawner.respawn_enemy(
                                current_time
                            )  # Handle respawn
                            break  # Only one hit can happen at a time

                    # Off-screen check (only if missile hasn't collided)
                    if not (
                        0 <= missile.pos["x"] <= self.game.screen_width
                        and 0 <= missile.pos["y"] <= self.game.screen_height
                    ):
                        self.finalize_missile_sequence(missile, success=False)
                        self.game.player.missiles.remove(missile)
                        del self.missile_lifespan[missile]
                        logging.debug("Training Mode: Missile left the screen.")

        if (
            self.game.is_respawning and current_time >= self.game.respawn_timer
        ):  # Correctly reference respawn timer in self.game
            if self.game.enemy:
                self.game.respawner.handle_respawn(
                    current_time
                )  # Correct call to handle respawn

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

            if self.game.data_logger:
                self.game.data_logger.log(frame_data)

        del self.missile_sequences[
            missile
        ]  # No need to check if key is in dict since it must be
        logging.debug(
            f"Finalized missile sequence with success={success}, frames={len(frames)}"
        )
