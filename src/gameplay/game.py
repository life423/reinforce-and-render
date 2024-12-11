import pygame
import random
from entities.player import Player
from entities.enemy import Enemy
from gameplay.menu import Menu
from core.data_logger import DataLogger
from gameplay.renderer import Renderer
from noise import pnoise1

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

        # Save data on exit
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
        self.enemy.update_movement()

        if self.check_collision():
            # Log collision if needed
            if self.mode == "train":
                self.data_logger.log({
                    "mode": "play",
                    "player_x": self.player.position["x"],
                    "player_y": self.player.position["y"],
                    "enemy_x": self.enemy.pos["x"],
                    "enemy_y": self.enemy.pos["y"],
                    "collision": True
                })
            else:
                print("Collision detected!")
                self.running = False

    def training_update(self):
        """Update logic for training mode."""
        self.player.noise_time += 0.01
        dx_player = pnoise1(self.player.noise_time + self.player.noise_offset_x) * self.player.step
        dy_player = pnoise1(self.player.noise_time + self.player.noise_offset_y) * self.player.step

        # Update player position with clamping
        self.player.position["x"] = max(0, min(SCREEN_WIDTH - self.player.size, self.player.position["x"] + dx_player))
        self.player.position["y"] = max(0, min(SCREEN_HEIGHT - self.player.size, self.player.position["y"] + dy_player))

        self.enemy.update_movement()
        collision = self.check_collision()

        # Log training data
        self.data_logger.log({
            "mode": "train",
            "player_x": self.player.position["x"],
            "player_y": self.player.position["y"],
            "enemy_x": self.enemy.pos["x"],
            "enemy_y": self.enemy.pos["y"],
            "collision": collision
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