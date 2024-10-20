import pygame
from noise import pnoise1
from core.db_connection import db  # Import database connection


class TrainingManager:
    def __init__(self, player, enemy, screen, data_logger):
        self.player = player
        self.enemy = enemy
        self.screen = screen
        self.data_logger = data_logger
        self.collection = db['training_data']  # Use your database collection

    def training_update(self):
        # Increment time to get new noise values for smooth movement
        self.player.noise_time += 0.01

        # Update player position using Perlin noise
        dx_player = pnoise1(self.player.noise_time +
                            self.player.noise_offset_x) * self.player.step
        dy_player = pnoise1(self.player.noise_time +
                            self.player.noise_offset_y) * self.player.step
        self.player.position["x"] = max(0, min(self.screen.get_width(
        ) - self.player.size, self.player.position["x"] + dx_player))
        self.player.position["y"] = max(0, min(self.screen.get_height(
        ) - self.player.size, self.player.position["y"] + dy_player))

        # Update enemy position
        self.enemy.update_movement()

        # Check for collision between player and enemy
        collision_occurred = self.check_collision()

        # Log the data to MongoDB
        training_data = {
            "timestamp": pygame.time.get_ticks(),
            "player_position": self.player.position,
            "enemy_position": self.enemy.pos,
            "distance": ((self.player.position["x"] - self.enemy.pos["x"]) ** 2 +
                         (self.player.position["y"] - self.enemy.pos["y"]) ** 2) ** 0.5,
            "collision": collision_occurred
        }
        # Insert training data into MongoDB
        self.collection.insert_one(training_data)

    def check_collision(self):
        player_rect = pygame.Rect(
            self.player.position["x"], self.player.position["y"], self.player.size, self.player.size)
        enemy_rect = pygame.Rect(
            self.enemy.pos["x"], self.enemy.pos["y"], self.enemy.size, self.enemy.size)

        return player_rect.colliderect(enemy_rect)
