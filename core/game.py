import pygame
from core.config import Config

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
        self.ENEMY_COLOR = Config.ENEMY_COLOR
        self.ENEMY_SIZE = Config.ENEMY_SIZE
        
        # Initialize Pygame and set up player position
        self.screen = self.initialize_pygame()
        self.SCREEN_WIDTH = self.screen.get_width()
        self.SCREEN_HEIGHT = self.screen.get_height()
        self.ENEMY_SPEED = max(1, self.SCREEN_WIDTH // 500)
        self.enemy_pos = {'x': 100, 'y': 300}
        self.enemy_direction = 1
        self.PLAYER_STEP = Config.PLAYER_STEP
        self.player_pos = {'x': self.SCREEN_WIDTH // 2, 'y': self.SCREEN_HEIGHT // 2}
        self.running = True
    
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

    def handle_player_movement_continuous(self, delta_time: float) -> None:
        """
        Continuously update the player's position based on pressed keys, taking into account the time elapsed to ensure smooth movement.
        The movement is restricted to ensure the player does not move off-screen.

        Args:
            delta_time (float): The time elapsed between the previous and current frame.
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
            pygame.K_s: (0, self.PLAYER_STEP)
        }

        for key, (dx, dy) in movement_directions.items():
            if keys[key]:
                new_x = self.player_pos['x'] + dx
                new_y = self.player_pos['y'] + dy
                # Ensure the player does not move off-screen
                self.player_pos['x'] = max(0, min(self.SCREEN_WIDTH - self.PLAYER_SIZE, new_x))
                self.player_pos['y'] = max(0, min(self.SCREEN_HEIGHT - self.PLAYER_SIZE, new_y))

    def update_enemy_position(self) -> None:
        """
        Update the enemy's position, making it patrol back and forth horizontally.
        The movement is restricted to ensure the enemy does not move off-screen.
        """
        self.enemy_pos['x'] += self.ENEMY_SPEED * self.enemy_direction
        # Reverse direction if the enemy hits the edge of the screen
        if self.enemy_pos['x'] <= 0 or self.enemy_pos['x'] + self.ENEMY_SIZE >= self.SCREEN_WIDTH:
            self.enemy_direction *= -1

    def draw_player(self) -> None:
        """
        Draw the player at the current position on the screen.
        The player is represented as a rectangle.
        """
        pygame.draw.rect(self.screen, self.PLAYER_COLOR, (self.player_pos['x'], self.player_pos['y'], self.PLAYER_SIZE, self.PLAYER_SIZE))

    def draw_enemy(self) -> None:
        """
        Draw the enemy at the current position on the screen.
        The enemy is represented as a rectangle.
        """
        pygame.draw.rect(self.screen, self.ENEMY_COLOR, (self.enemy_pos['x'], self.enemy_pos['y'], self.ENEMY_SIZE, self.ENEMY_SIZE))

    def run(self) -> None:
        """
        Main game loop that runs the game.
        Handles events, updates game state, and redraws the screen.
        """
        clock = pygame.time.Clock()
        while self.running:
            delta_time = clock.tick(60) / 1000.0  # Calculate delta time in seconds
            self.handle_events()
            self.handle_player_movement_continuous(delta_time)
            self.update_enemy_position()
            self.screen.fill(self.BACKGROUND_COLOR)
            self.draw_player()
            self.draw_enemy()
            pygame.display.flip()
        pygame.quit()