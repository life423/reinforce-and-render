import sys
import os
import pygame
import json
import random

from core.config import Config
from core.utils import clamp_position  # Add this import
from entities.enemy import Enemy

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class Game:
    def __init__(self, config: Config) -> None:
        """
        Initialize the game, including Pygame and game variables like player position.
        """
        # Game configuration
        self.SCREEN_TITLE = config.SCREEN_TITLE
        self.BACKGROUND_COLOR = config.BACKGROUND_COLOR
        self.PLAYER_COLOR = config.PLAYER_COLOR
        self.PLAYER_SIZE = config.PLAYER_SIZE

        # Initialize Pygame and set up player position
        self.screen = self.initialize_pygame()
        self.SCREEN_WIDTH = self.screen.get_width()
        self.SCREEN_HEIGHT = self.screen.get_height()

        # Enemy instance
        self.enemy = Enemy(
            start_x=100,
            start_y=300,
            screen_width=self.SCREEN_WIDTH,
            screen_height=self.SCREEN_HEIGHT,
        )

        self.PLAYER_STEP = config.PLAYER_STEP
        self.player_pos = {"x": self.SCREEN_WIDTH // 2, "y": self.SCREEN_HEIGHT // 2}
        self.running = True

        # List to store collision data for training
        self.collision_data = []

        # Frame counter for player movement
        self.player_movement_counter = 0
        self.player_current_direction = random.choice(
            [
                (self.PLAYER_STEP, 0),
                (-self.PLAYER_STEP, 0),
                (0, self.PLAYER_STEP),
                (0, -self.PLAYER_STEP),
            ]
        )

        # Set up data directory and file path
        self.data_dir, self.collision_data_file = self.setup_data_directory()
        self.collision_data_file = os.path.join(self.data_dir, "collision_data.json")
        print(f"Collision data will be saved to: {self.collision_data_file}")

        # Clear previous collision data file if it exists
        if os.path.exists(self.collision_data_file):
            os.remove(self.collision_data_file)

            # Set up data directory and file
        self.data_dir, self.collision_data_file = self.setup_data_directory()

    def setup_data_directory(self) -> tuple[str, str]:
        """
        Set up the data directory and return paths for directory and collision data file.
        Returns:
            Tuple[str, str]: The data directory path and the collision data file path.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        collision_data_file = os.path.join(data_dir, "collision_data.json")
        print(f"Collision data will be saved to: {collision_data_file}")

        # Clear previous collision data file if it exists
        if os.path.exists(collision_data_file):
            os.remove(collision_data_file)

        return data_dir, collision_data_file
        self.player_movement_counter = 0
        self.player_current_direction = random.choice(
            [
                (self.PLAYER_STEP, 0),
                (-self.PLAYER_STEP, 0),
                (0, self.PLAYER_STEP),
                (0, -self.PLAYER_STEP),
            ]
        )

    def initialize_pygame(self) -> pygame.Surface:
        """
        Initialize all imported Pygame modules and set up the display in fullscreen mode.

        Returns:
            pygame.Surface: The screen surface to draw on.
        """
        pygame.init()
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption(self.SCREEN_TITLE)
        return screen

    def handle_events(self) -> None:
        """
        Listen for events such as quitting the game and player movement.
        Updates the running state and player's position accordingly.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    self.running = False

    def handle_player_movement_random(self) -> None:
        """
        Update the player's position in a more varied, dynamic manner.
        """
        # Change direction every 20 frames
        if self.player_movement_counter % 20 == 0:
            self.player_current_direction = random.choice(
                [
                    (self.PLAYER_STEP, 0),
                    (-self.PLAYER_STEP, 0),
                    (0, self.PLAYER_STEP),
                    (0, -self.PLAYER_STEP),
                ]
            )

        # Apply the movement
        dx, dy = self.player_current_direction
        new_x = self.player_pos["x"] + dx
        new_y = self.player_pos["y"] + dy

        # Ensure the player does not move off-screen
        self.player_pos["x"] = clamp_position(
            new_x, 0, self.SCREEN_WIDTH - self.PLAYER_SIZE
        )
        self.player_pos["y"] = clamp_position(
            new_y, 0, self.SCREEN_HEIGHT - self.PLAYER_SIZE
        )

        # Increment movement counter
        self.player_movement_counter += 1

    def handle_player_movement_manual(self) -> None:
        """
        Continuously update the player's position based on pressed keys.
        The movement is restricted to ensure the player does not move off-screen.
        """
        keys = pygame.key.get_pressed()
        movement_directions = {
            pygame.K_LEFT: (-self.PLAYER_STEP, 0),
            pygame.K_a: (-self.PLAYER_STEP, 0),
            pygame.K_RIGHT: (self.PLAYER_STEP, 0),
            pygame.K_d: (self.PLAYER_STEP, 0),
            pygame.K_UP: (0, -self.PLAYER_STEP),
            pygame.K_w: (0, -self.PLAYER_STEP),
            pygame.K_DOWN: (0, self.PLAYER_STEP),
            pygame.K_s: (0, self.PLAYER_STEP),
        }

        for key, (dx, dy) in movement_directions.items():
            if keys[key]:
                new_x = self.player_pos["x"] + dx
                new_y = self.player_pos["y"] + dy
                # Ensure the player does not move off-screen
                self.player_pos["x"] = clamp_position(
                    new_x, 0, self.SCREEN_WIDTH - self.PLAYER_SIZE
                )
                self.player_pos["y"] = clamp_position(
                    new_y, 0, self.SCREEN_HEIGHT - self.PLAYER_SIZE
                )

    def log_game_state(self) -> None:
        """
        Log the current game state to be used for training.
        """
        self.collision_data.append(
            {
                "player_position": self.player_pos.copy(),
                "enemy_position": self.enemy.pos.copy(),
                "time": pygame.time.get_ticks(),
                "distance": (
                    (self.player_pos["x"] - self.enemy.pos["x"]) ** 2
                    + (self.player_pos["y"] - self.enemy.pos["y"]) ** 2
                )
                ** 0.5,
            }
        )

        # Save every 100 frames to avoid performance hit
        if len(self.collision_data) % 100 == 0:
            print("Saving collision data...")
            self.save_collision_data()

    def check_collision(self) -> bool:
        """
        Check for collision between the player and the enemy.

        Returns:
            bool: True if a collision is detected, otherwise False.
        """
        player_rect = pygame.Rect(
            self.player_pos["x"],
            self.player_pos["y"],
            self.PLAYER_SIZE,
            self.PLAYER_SIZE,
        )
        enemy_rect = pygame.Rect(
            self.enemy.pos["x"], self.enemy.pos["y"], self.enemy.size, self.enemy.size
        )
        return player_rect.colliderect(enemy_rect)

    def save_collision_data(self) -> None:
        """
        Save the collision data to a JSON file.
        """
        try:
            with open(self.collision_data_file, "w") as file:
                json.dump(self.collision_data, file, indent=4)
            print(f"Collision data saved successfully to {self.collision_data_file}.")
        except Exception as e:
            print(f"Failed to save collision data: {e}")

    def run(self, mode="training") -> None:
        """
        Main game loop that runs the game.
        Handles events, updates game state, and redraws the screen.
        """
        clock = pygame.time.Clock()

        # Assign movement handlers based on mode
        player_movement_handler = (
            self.handle_player_movement_random
            if mode == "training"
            else self.handle_player_movement_manual
        )
        enemy_movement_handler = (
            self.enemy.update_position_randomly
            if mode == "training"
            else lambda: self.enemy.update_position(self.player_pos)
        )

        while self.running:
            delta_time = clock.tick(60) / 1000.0  # Calculate delta time in seconds
            self.handle_events()

            # Handle player and enemy movement
            player_movement_handler()
            enemy_movement_handler()

            # Log game state for training purposes
            self.log_game_state()

            # Check collision
            if self.check_collision():
                self.log_game_state()

            # Draw everything
            self.screen.fill(self.BACKGROUND_COLOR)
            self.draw_player()
            self.enemy.draw(self.screen)
            pygame.display.flip()

        # Save remaining collision data on game exit
        if self.collision_data:
            self.save_collision_data()

        pygame.quit()

    def draw_player(self) -> None:
        """
        Draw the player at the current position on the screen.
        The player is represented as a rectangle.
        """
        pygame.draw.rect(
            self.screen,
            self.PLAYER_COLOR,
            (
                self.player_pos["x"],
                self.player_pos["y"],
                self.PLAYER_SIZE,
                self.PLAYER_SIZE,
            ),
        )
