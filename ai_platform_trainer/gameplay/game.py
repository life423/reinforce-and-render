import pygame
import math
from noise import pnoise1
from ai_platform_trainer.entities.player import Player
from ai_platform_trainer.entities.enemy import Enemy
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.gameplay.renderer import Renderer
from ai_platform_trainer.core.data_logger import DataLogger

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WINDOW_TITLE = "Pixel Pursuit"
FRAME_RATE = 60
DATA_PATH = "data/training_data.json"


class Game:
    """Main class to run the Pixel Pursuit game."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()

        # Entities and managers
        self.player = Player(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.enemy = Enemy(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.menu = Menu(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.renderer = Renderer(self.screen)
        self.data_logger = DataLogger(DATA_PATH)

        # Game states
        self.running = True
        self.menu_active = True
        self.mode = None  # "train" or "play"

    def run(self):
        """Main game loop."""
        while self.running:
            self.handle_events()

            if self.menu_active:
                self.menu.draw(self.screen)
            else:
                self.update()
                self.renderer.render(self.menu, self.player, self.enemy, self.menu_active, self.screen)

            pygame.display.flip()
            self.clock.tick(FRAME_RATE)

        # Only save if in training mode
        if self.mode == "train":
            self.data_logger.save()

        pygame.quit()

    def handle_events(self):
        """Handle all window and menu-related events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
            elif self.menu_active:
                selected_action = self.menu.handle_menu_events(event)
                if selected_action:
                    self.check_menu_selection(selected_action)

    def check_menu_selection(self, selected_action: str) -> None:
        """Handle actions selected from the menu."""
        if selected_action == "exit":
            self.running = False
        elif selected_action in ["train", "play"]:
            self.menu_active = False
            self.start_game(selected_action)

    def start_game(self, mode: str) -> None:
        """Initialize game entities and set mode."""
        self.mode = mode
        print(mode)
        self.player.reset()
        self.enemy.reset()

        # Place player and enemy at distinct positions
        self.player.position["x"] = SCREEN_WIDTH // 4 - self.player.size // 2
        self.player.position["y"] = SCREEN_HEIGHT // 2 - self.player.size // 2
        self.enemy.pos["x"] = (SCREEN_WIDTH * 3) // 4 - self.enemy.size // 2
        self.enemy.pos["y"] = SCREEN_HEIGHT // 2 - self.enemy.size // 2

    def update(self):
        """Update game state depending on the mode."""
        if self.mode == "train":
            self.training_update()
        elif self.mode == "play":
            self.play_update()

    def check_collision(self) -> bool:
        """Check if the player and enemy collide."""
        player_rect = pygame.Rect(
            self.player.position["x"], self.player.position["y"], self.player.size, self.player.size
        )
        enemy_rect = pygame.Rect(
            self.enemy.pos["x"], self.enemy.pos["y"], self.enemy.size, self.enemy.size
        )
        return player_rect.colliderect(enemy_rect)

    def play_update(self):
        """Update logic for play mode."""
        self.handle_input()
        # Pass player's x, y, and speed to enemy.update_movement()
        self.enemy.update_movement(self.player.position["x"], self.player.position["y"], self.player.step)

        if self.check_collision():
            print("Collision detected!")
            self.running = False

    def training_update(self):
        """Update logic for training mode."""
        # Calculate distance before movement
        dist_before = math.dist(
            (self.enemy.pos["x"], self.enemy.pos["y"]),
            (self.player.position["x"], self.player.position["y"])
        )

        # Move player using its own update method (noise-based movement)
        self.player.update()

        # Move enemy towards the player
        self.enemy.update_movement(
            self.player.position["x"], 
            self.player.position["y"], 
            self.player.step
        )

        # Calculate distance after movement
        dist_after = math.dist(
            (self.enemy.pos["x"], self.enemy.pos["y"]),
            (self.player.position["x"], self.player.position["y"])
        )

        collision = self.check_collision()

        # Calculate reward
        reward = (dist_before - dist_after) * 10
        reward -= 1
        if collision:
            reward += 100

        # Log training data
        self.data_logger.log({
            "mode": "train",
            "player_x": self.player.position["x"],
            "player_y": self.player.position["y"],
            "enemy_x": self.enemy.pos["x"],
            "enemy_y": self.enemy.pos["y"],
            "distance_before": dist_before,
            "distance_after": dist_after,
            "collision": collision,
            "reward": reward
        })

    def handle_input(self):
        """Handle player keyboard input for movement."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.player.position['x'] -= self.player.step
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.player.position['x'] += self.player.step
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.player.position['y'] -= self.player.step
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.player.position['y'] += self.player.step
        if keys[pygame.K_ESCAPE]:
            self.running = False

        # Clamp player position
        self.player.position['x'] = max(0, min(self.player.position['x'], SCREEN_WIDTH - self.player.size))
        self.player.position['y'] = max(0, min(self.player.position['y'], SCREEN_HEIGHT - self.player.size))

if __name__ == "__main__":
    game = Game()
    game.run()