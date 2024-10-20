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
            if self.menu_active:
                # Only draw the menu if the menu is active
                self.menu.draw(self.screen)
            else:
                # Draw and update game logic otherwise
                self.update()
                self.renderer.render(self.menu, self.player,
                                     self.enemy, self.menu_active, self.screen)

            pygame.display.flip()
            self.clock.tick(60)  # Cap the frame rate

        # Save data on exit
        self.data_manager.save_collision_data(self.collision_data)
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif self.menu_active:
                selected_action = self.menu.handle_menu_events(event)
                if selected_action:
                    self.check_menu_selection(selected_action)
            else:
                # Add additional event handling for gameplay if needed
                pass

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
        # Training mode logic (for now, player and enemy move randomly)
        self.player.update()  # Player moves manually
        # In training mode, enemy moves in a more complex way for learning
        self.enemy.update(self.player.get_position())
