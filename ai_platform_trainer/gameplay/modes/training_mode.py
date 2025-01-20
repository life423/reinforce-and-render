import math
import logging
import random
import pygame
from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.utils.helpers import wrap_position
from ai_platform_trainer.gameplay.utils import compute_normalized_direction


class TrainingMode:
    def __init__(self, game):
        """
        Manages training-specific logic, including spawning missiles, logging data,
        and finalizing missile sequences for further AI model training.
        """
        self.game = game
        self.missile_cooldown = 0
        # Dictionary mapping missile -> (birth_time, lifespan)
        self.missile_lifespan = {}
        # Dictionary mapping missile -> list of frame-data dictionaries
        self.missile_sequences = {}

    def update(self):
        """
        Called once per frame during training mode.
        Handles:
        - Player and enemy updates
        - Random missile firing logic
        - Missile collisions, lifespan checks
        - Logging per-frame data to missile sequences
        """
        current_time = pygame.time.get_ticks()

        # If both enemy and player exist, proceed with training updates
        if self.game.enemy and self.game.player:
            enemy_x = self.game.enemy.pos["x"]
            enemy_y = self.game.enemy.pos["y"]

            # 1) Update player and enemy
            self.game.player.update(enemy_x, enemy_y)
            self.game.enemy.update_movement(
                self.game.player.position["x"],
                self.game.player.position["y"],
                self.game.player.step,
            )

            # 2) Handle missile cooldown and random firing chance
            if self.missile_cooldown > 0:
                self.missile_cooldown -= 1

            # Default missile_fire_prob = 0.02 (2% chance each frame)
            if random.random() < getattr(self, "missile_fire_prob", 0.02):
                # Only fire if cooldown is done and no missiles are active
                if self.missile_cooldown <= 0 and len(self.game.player.missiles) == 0:
                    self.game.player.shoot_missile(enemy_x, enemy_y)
                    self.missile_cooldown = (
                        120  # Reset cooldown (e.g., 2 seconds @ 60 FPS)
                    )

                    if self.game.player.missiles:
                        missile = self.game.player.missiles[0]
                        # Store lifespan info for this missile
                        self.missile_lifespan[missile] = (
                            missile.birth_time,
                            missile.lifespan,
                        )
                        # Initialize an empty list for collecting frame data
                        self.missile_sequences[missile] = []

        # 3) Check missile collisions & lifespans
        current_missiles = self.game.player.missiles[:]
        for missile in current_missiles:
            if missile in self.missile_lifespan:
                birth_time, lifespan = self.missile_lifespan[missile]
                # If missile's time is up, finalize sequence
                if current_time - birth_time >= lifespan:
                    self.finalize_missile_sequence(missile, success=False)
                    self.game.player.missiles.remove(missile)
                    del self.missile_lifespan[missile]
                    logging.debug("Training mode: Missile removed (lifespan expiry).")
                else:
                    # Update missile position
                    missile.update()

                    # ----------------------
                    # NEW SECTION: Log data each frame
                    # ----------------------
                    if missile in self.missile_sequences:
                        player_x = self.game.player.position["x"]
                        player_y = self.game.player.position["y"]
                        enemy_x = self.game.enemy.pos["x"]
                        enemy_y = self.game.enemy.pos["y"]

                        dist_player_enemy = math.hypot(
                            enemy_x - player_x, enemy_y - player_y
                        )
                        dist_missile_enemy = math.hypot(
                            enemy_x - missile.pos["x"], enemy_y - missile.pos["y"]
                        )

                        self.missile_sequences[missile].append(
                            {
                                "pos_x": missile.pos["x"],
                                "pos_y": missile.pos["y"],
                                "timestamp": current_time,
                                "player_x": player_x,
                                "player_y": player_y,
                                "enemy_x": enemy_x,
                                "enemy_y": enemy_y,
                                "dist_player_enemy": dist_player_enemy,
                                "dist_missile_enemy": dist_missile_enemy,
                                # Add more fields as desired
                            }
                        )
                    # End new section

                    # 4) Collision check with enemy
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

                            self.game.enemy.hide()
                            self.game.is_respawning = True
                            self.game.respawn_timer = (
                                current_time + self.game.respawn_delay
                            )
                            # Only handle one collision this frame
                            break

                    # 5) Off-screen check
                    if not (
                        0 <= missile.pos["x"] <= self.game.screen_width
                        and 0 <= missile.pos["y"] <= self.game.screen_height
                    ):
                        self.finalize_missile_sequence(missile, success=False)
                        self.game.player.missiles.remove(missile)
                        del self.missile_lifespan[missile]
                        logging.debug("Training Mode: Missile left the screen.")

        # 6) Enemy respawn check
        if self.game.is_respawning and current_time >= self.game.respawn_timer:
            if self.game.enemy:
                self.game.handle_respawn(current_time)

    def finalize_missile_sequence(self, missile, success: bool) -> None:
        """
        Called when a missile's life ends, either by expiration, collision, or going off-screen.
        Logs all frame data with a missile_collision outcome, then removes missile sequence data.
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
