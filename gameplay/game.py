import pygame
from entities.player import Player
from entities.enemy import Enemy
from gameplay.menu import Menu
from gameplay.renderer import Renderer
from gameplay.data_manager import DataManager


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
        self.data_manager = DataManager("data/collision_data.json")
        self.collision_data = self.data_manager.load_collision_data()

        # Game states
        self.running = True
        self.menu_active = True
        self.mode = None  # Training or Play

    def run(self):
        while self.running:
            self.handle_events()
            if not self.menu_active:
                self.update()
            self.renderer.render(self.menu, self.player,
                                 self.enemy, self.menu_active, self.screen)
            self.clock.tick(60)  # Cap the frame rate

        # Save data on exit
        self.data_manager.save_collision_data(self.collision_data)
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif self.menu_active:
                self.menu.handle_menu_events(event)
                self.check_menu_selection()

    def check_menu_selection(self):
        selected_option = self.menu.menu_options[self.menu.selected_option].lower(
        )
        if selected_option == "exit":
            self.running = False
        elif selected_option in ["training", "play"]:
            self.menu_active = False
            self.start_game(selected_option)

    def start_game(self, mode: str):
        self.mode = mode
        self.player.reset()
        self.enemy.reset()

    def update(self):
        # Update player and enemy movement
        self.player.update()  # Update player position based on input
        # Update enemy position based on AI logic
        self.enemy.update(self.player.get_position())



