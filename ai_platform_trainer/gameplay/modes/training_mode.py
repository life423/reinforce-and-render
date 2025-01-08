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
        self.missile_lifespan = {}
        self.missile_sequences = {}

    def update(self):
        """Handles updates specifically for training mode."""
        current_time = pygame.time.get_ticks()

        if self.game.enemy and self.game.player:
            enemy_x = self.game.enemy.pos["x"]
            enemy_y = self.game.enemy.pos["y"]

            self.game.player.update(enemy_x, enemy_y)
            self.game.enemy.update_movement(
                self.game.player.position["x"],
                self.game.player.position["y"],
                self.game.player.step,
            )

            if random.random() < getattr(self, "missile_fire_prob", 0.02):
                if self.missile_cooldown > 0:
                    self.missile_cooldown -= 1

                if self.missile_cooldown <= 0 and len(self.game.player.missiles) == 0:
                    self.game.player.shoot_missile(enemy_x, enemy_y)
                    self.missile_cooldown = 120

                    if self.game.player.missiles:
                        msl = self.game.player.missiles[0]
                        self.missile_lifespan[msl] = (msl.birth_time, msl.lifespan)
                        self.missile_sequences[msl] = []

        for missile in self.game.player.missiles[:]:
            if missile in self.missile_lifespan:
                birth_time, lifespan = self.missile_lifespan[missile]
                if current_time - birth_time >= lifespan:
                    self.finalize_missile_sequence(missile, success=False)
                    self.game.player.missiles.remove(missile)
                    del self.missile_lifespan[missile]
                    logging.debug("Training mode: Missile removed (lifespan expiry).")
                else:
                    missile.update()

                if self.game.enemy:
                    enemy_rect = pygame.Rect(
                        self.game.enemy.pos["x"],
                        self.game.enemy.pos["y"],
                        self.game.enemy.size,
                        self.game.enemy.size,
                    )

                    if missile.get_rect().colliderect(enemy_rect):
                        logging.info("Missile hit the enemy (training mode).")
                        self.finalize_missile_sequence(missile, success=True)
                        self.game.player.missiles.remove(missile)

                        del self.missile_lifespan[missile]
                        self.game.respawner.respawn_enemy(current_time)
                        break

                    if not (
                        0 <= missile.pos["x"] <= self.game.screen_width
                        and 0 <= missile.pos["y"] <= self.game.screen_height
                    ):
                        self.finalize_missile_sequence(missile, success=False)
                        self.game.player.missiles.remove(missile)
                        del self.missile_lifespan[missile]
                        logging.debug("Training Mode: Missile left the screen.")

        if self.game.is_respawning and current_time >= self.game.respawn_timer:
            if self.game.enemy:
                self.game.respawner.handle_respawn(current_time)

    def finalize_missile_sequence(self, missile, success: bool) -> None:
        """
        Mark all frames in missile's sequence with 'missile_collision'= success
        Then log them to self.game.data_logger, and clear from missile_sequences.
        """
        if missile not in self.missile_sequences:
            return

        outcome_val = success
        frames = self.missile_sequences[missile]

        for frame_data in frames:
            frame_data["missile_collision"] = outcome_val

            if self.game.data_logger:
                self.game.data_logger.log(frame_data)

        del self.missile_sequences[missile]
        logging.debug(
            f"Finalized missile sequence with success={success}, frames={len(frames)}"
        )
