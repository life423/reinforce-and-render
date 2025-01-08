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
        current_time = pygame.time.get_ticks()

        if self.game.enemy and self.game.player:
            enemy_x = self.game.enemy.pos["x"]
            enemy_y = self.game.enemy.pos["y"]

            # Player & Enemy updates
            self.game.player.update(enemy_x, enemy_y)
            self.game.enemy.update_movement(
                self.game.player.position["x"],
                self.game.player.position["y"],
                self.game.player.step,
            )

            # Missile cooldown & random chance
            if self.missile_cooldown > 0:
                self.missile_cooldown -= 1

            if random.random() < getattr(self, "missile_fire_prob", 0.02):
                if self.missile_cooldown <= 0 and len(self.game.player.missiles) == 0:
                    self.game.player.shoot_missile(enemy_x, enemy_y)
                    self.missile_cooldown = 120

                    if self.game.player.missiles:
                        missile = self.game.player.missiles[0]
                        self.missile_lifespan[missile] = (
                            missile.birth_time,
                            missile.lifespan,
                        )
                        self.missile_sequences[missile] = []

        # Check missile collisions, lifespan
        current_missiles = self.game.player.missiles[:]
        for missile in current_missiles:
            if missile in self.missile_lifespan:
                birth_time, lifespan = self.missile_lifespan[missile]
                if current_time - birth_time >= lifespan:
                    self.finalize_missile_sequence(missile, success=False)
                    self.game.player.missiles.remove(missile)
                    del self.missile_lifespan[missile]
                    logging.debug("Training mode: Missile removed (lifespan expiry).")
                else:
                    missile.update()

                    # Collision with enemy
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

                            # <-- Instead of self.game.respawner.respawn_enemy(current_time):
                            # do the same approach as "play" mode
                            self.game.enemy.hide()
                            self.game.is_respawning = True
                            self.game.respawn_timer = (
                                current_time + self.game.respawn_delay
                            )

                            break  # only handle one collision per frame

                    # Off-screen check
                    if not (
                        0 <= missile.pos["x"] <= self.game.screen_width
                        and 0 <= missile.pos["y"] <= self.game.screen_height
                    ):
                        self.finalize_missile_sequence(missile, success=False)
                        self.game.player.missiles.remove(missile)
                        del self.missile_lifespan[missile]
                        logging.debug("Training Mode: Missile left the screen.")

        # If it's time to actually respawn the enemy
        if self.game.is_respawning and current_time >= self.game.respawn_timer:
            if self.game.enemy:
                # Just let your Game.handle_respawn(...) or similar method do it:
                self.game.handle_respawn(current_time)

    def finalize_missile_sequence(self, missile, success: bool) -> None:
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
