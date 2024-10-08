import pygame
import json
import os
import random
from core.config import Config
from entities.enemy import Enemy

class Game:
    def __init__(self) -> None:
        """
        Initialize the game, including Pygame and game variables like player position.
        """
        # Game configuration
        self.SCREEN_TITLE = Config.SCREEN_TITLE
        self.BACKGROUND_COLOR = Config.BACKGROUND_COLOR
        self.PLAYER_COLOR = Config.PLAYER_COLOR
        self.PLAYER_SIZE = Config.PLAYER_SIZE

        # Initialize Pygame and set up player position
        self.screen = self.initialize_pygame()
        self.SCREEN_WIDTH = self.screen.get_width()
        self.SCREEN_HEIGHT = self.screen.get_height()

        # Enemy instance
        self.enemy = Enemy(start_x=100, start_y=300, screen_width=self.SCREEN_WIDTH, screen_height=self.SCREEN_HEIGHT)

        self.PLAYER_STEP = Config.PLAYER_STEP
        self.player_pos = {'x': self.SCREEN_WIDTH // 2, 'y': self.SCREEN_HEIGHT // 2}
        self.running = True

        # List to store collision data for training
        self.collision_data = []

        # Clear previous collision data file if it exists
        if os.path.exists("collision_data.json"):
            os.remove("collision_data.json")

        # Frame counter for player movement
        self.player_movement_counter = 0
        self.player_current_direction = random.choice([(self.PLAYER_STEP, 0), (-self.PLAYER_STEP, 0), (0, self.PLAYER_STEP), (0, -self.PLAYER_STEP)])
    
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
            self.player_current_direction = random.choice([
                (self.PLAYER_STEP, 0), (-self.PLAYER_STEP, 0), (0, self.PLAYER_STEP), (0, -self.PLAYER_STEP)
            ])
    
        # Apply the movement
        dx, dy = self.player_current_direction
        new_x = self.player_pos['x'] + dx
        new_y = self.player_pos['y'] + dy

        # Ensure the player does not move off-screen
        self.player_pos['x'] = max(0, min(self.SCREEN_WIDTH - self.PLAYER_SIZE, new_x))
        self.player_pos['y'] = max(0, min(self.SCREEN_HEIGHT - self.PLAYER_SIZE, new_y))

        # Increment movement counter
        self.player_movement_counter += 1

    def log_game_state(self) -> None:
        """
        Log the current game state to be used for training.
        """
        self.collision_data.append({
            'player_position': self.player_pos.copy(),
            'enemy_position': self.enemy.pos.copy(),
            'time': pygame.time.get_ticks(),
            'distance': ((self.player_pos['x'] - self.enemy.pos['x'])**2 + (self.player_pos['y'] - self.enemy.pos['y'])**2) ** 0.5
        })
        
        # Save every 100 frames to avoid performance hit
        if len(self.collision_data) % 100 == 0:
            self.save_collision_data()

    def check_collision(self) -> bool:
        """
        Check for collision between the player and the enemy.

        Returns:
            bool: True if a collision is detected, otherwise False.
        """
        player_rect = pygame.Rect(self.player_pos['x'], self.player_pos['y'], self.PLAYER_SIZE, self.PLAYER_SIZE)
        enemy_rect = pygame.Rect(self.enemy.pos['x'], self.enemy.pos['y'], self.enemy.size, self.enemy.size)
        return player_rect.colliderect(enemy_rect)

    def save_collision_data(self) -> None:
        """
        Save the collision data to a JSON file.
        """
        with open("collision_data.json", "w") as file:
            json.dump(self.collision_data, file, indent=4)

    def run(self) -> None:
        """
        Main game loop that runs the game.
        Handles events, updates game state, and redraws the screen.
        """
        clock = pygame.time.Clock()
        while self.running:
            delta_time = clock.tick(60) / 1000.0  # Calculate delta time in seconds
            self.handle_events()
            self.handle_player_movement_random()  # Use random movement for the player
            self.enemy.update_position(self.player_pos)

            # Log game state for training purposes
            self.log_game_state()

            # Check collision
            if self.check_collision():
                print("Collision Detected! Logging data...")
                self.log_game_state()
                # Instead of stopping the game, continue running

            # Draw everything
            self.screen.fill(self.BACKGROUND_COLOR)
            self.draw_player()
            self.enemy.draw(self.screen)
            pygame.display.flip()
        pygame.quit()

    def draw_player(self) -> None:
        """
        Draw the player at the current position on the screen.
        The player is represented as a rectangle.
        """
        pygame.draw.rect(self.screen, self.PLAYER_COLOR, (self.player_pos['x'], self.player_pos['y'], self.PLAYER_SIZE, self.PLAYER_SIZE))