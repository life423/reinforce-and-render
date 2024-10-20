import pygame
import random
from entities.player import Player
from entities.enemy import Enemy
from gameplay.menu import Menu
from gameplay.renderer import Renderer
from core.data_logger import DataLogger


class Game:
    def __init__(self):
        # Initialize screen and clock
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Pixel Pursuit")
        self.clock = pygame.time.Clock()

        # Initialize entities and managers
        self.player = Player(self.screen.get_width(), self.screen.get_height())
        self.enemy = Enemy(self.screen.get_width(), self.screen.get_height())
        self.menu = Menu(self.screen.get_width(), self.screen.get_height())
        self.renderer = Renderer(self.screen)
        self.data_logger = DataLogger("data/training_data.json")

        # Game states
        self.running = True
        self.menu_active = True
        self.mode = None  # Training or Play

    def run(self):
        while self.running:
            self.handle_events()
            if self.menu_active:
                self.menu.draw(self.screen)
            else:
                self.update()
                self.renderer.render(self.menu, self.player,
                                     self.enemy, self.menu_active, self.screen)

            pygame.display.flip()
            self.clock.tick(60)  # Cap the frame rate

        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif self.menu_active:
                selected_action = self.menu.handle_menu_events(event)
                if selected_action:
                    self.check_menu_selection(selected_action)

    def check_menu_selection(self, selected_action):
        if selected_action == "exit":
            self.running = False
        elif selected_action in ["train", "play"]:
            self.menu_active = False
            self.start_game(selected_action)

    def start_game(self, mode: str):
        self.mode = mode
        self.player.reset()
        self.enemy.reset()

    def update(self):
        if self.mode == "train":
            self.training_update()
        elif self.mode == "play":
            self.play_update()

    def play_update(self):
        # Update player and enemy movement for play mode
        self.player.update()
        self.enemy.update(self.player.get_position())

    def training_update(self):
        # Update player and enemy randomly for training mode
        dx = random.choice([-1, 0, 1]) * self.player.step
        dy = random.choice([-1, 0, 1]) * self.player.step
        self.player.position["x"] = max(0, min(
            self.screen.get_width() - self.player.size, self.player.position["x"] + dx))
        self.player.position["y"] = max(0, min(
            self.screen.get_height() - self.player.size, self.player.position["y"] + dy))

        dx = random.choice([-1, 0, 1]) * self.enemy.speed
        dy = random.choice([-1, 0, 1]) * self.enemy.speed
        self.enemy.pos["x"] = max(
            0, min(self.screen.get_width() - self.enemy.size, self.enemy.pos["x"] + dx))
        self.enemy.pos["y"] = max(
            0, min(self.screen.get_height() - self.enemy.size, self.enemy.pos["y"] + dy))

        # Log training data using DataLogger
        self.data_logger.log_data(self.player.position, self.enemy.pos)
