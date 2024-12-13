import pygame
import math
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.player_training import Player as PlayerTraining
from ai_platform_trainer.entities.enemy_play import Enemy as EnemyPlay
from ai_platform_trainer.entities.enemy_training import Enemy as EnemyTrain
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

        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT

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
                self.renderer.render(
                    self.menu, self.player, self.enemy, self.menu_active, self.screen
                )

            pygame.display.flip()
            self.clock.tick(FRAME_RATE)

        # Only save if in training mode
        if self.mode == "train":
            self.data_logger.save()

        pygame.quit()

    def handle_events(self):
        """Handle all window and menu-related events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
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
        self.mode = mode
        print(mode)

        if mode == "train":
            self.player = PlayerTraining(self.screen_width, self.screen_height)
            self.enemy = EnemyTrain(self.screen_width, self.screen_height)
        else:
            self.player = PlayerPlay(self.screen_width, self.screen_height)
            self.enemy = EnemyPlay(self.screen_width, self.screen_height)

        # Position entities
        self.player.reset()
        self.player.position["x"] = self.screen_width // 4 - self.player.size // 2
        self.player.position["y"] = self.screen_height // 2 - self.player.size // 2
        self.enemy.pos["x"] = (self.screen_width * 3) // 4 - self.enemy.size // 2
        self.enemy.pos["y"] = self.screen_height // 2 - self.enemy.size // 2

    def update(self):
        """Update game state depending on the mode."""
        if self.mode == "train":
            self.training_update()
        elif self.mode == "play":
            self.play_update()

    def check_collision(self) -> bool:
        """Check if the player and enemy collide."""
        player_rect = pygame.Rect(
            self.player.position["x"],
            self.player.position["y"],
            self.player.size,
            self.player.size,
        )
        enemy_rect = pygame.Rect(
            self.enemy.pos["x"], self.enemy.pos["y"], self.enemy.size, self.enemy.size
        )
        return player_rect.colliderect(enemy_rect)

    def play_update(self):
        """Update logic for play mode."""
        # Let the player handle input directly
        if not self.player.handle_input():
            self.running = False
            return

        # Enemy moves and collision check
        self.enemy.update_movement(
            self.player.position["x"], self.player.position["y"], self.player.step
        )
        if self.check_collision():
            print("Collision detected!")
            self.running = False

    def training_update(self):
        """Update logic for training mode."""
        self.player.update(self.enemy.pos["x"], self.enemy.pos["y"])
        old_enemy_x = self.enemy.pos["x"]
        old_enemy_y = self.enemy.pos["y"]
        self.enemy.update_movement(
            self.player.position["x"], self.player.position["y"], self.player.step
        )

        action_dx = self.enemy.pos["x"] - old_enemy_x
        action_dy = self.enemy.pos["y"] - old_enemy_y
        collision = self.check_collision()

        self.data_logger.log(
            {
                "mode": "train",
                "player_x": self.player.position["x"],
                "player_y": self.player.position["y"],
                "enemy_x": self.enemy.pos["x"],
                "enemy_y": self.enemy.pos["y"],
                "action_dx": action_dx,
                "action_dy": action_dy,
                "collision": collision,
            }
        )


if __name__ == "__main__":
    game = Game()
    game.run()
