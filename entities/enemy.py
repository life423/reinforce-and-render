import pygame
from core.config import Config

class Enemy:
    def __init__(self, start_x: int, start_y: int, screen_width: int) -> None:
        """
        Initialize the enemy with position, size, color, and movement settings.

        Args:
            start_x (int): Initial x position of the enemy.
            start_y (int): Initial y position of the enemy.
            screen_width (int): The width of the screen for restricting movement.
        """
        self.pos = {'x': start_x, 'y': start_y}
        self.size = Config.ENEMY_SIZE
        self.color = Config.ENEMY_COLOR
        self.speed = max(1, screen_width // 500)  # Dynamic speed based on screen width
        self.direction = 1  # 1 for right, -1 for left
        self.screen_width = screen_width

    def update_position(self) -> None:
        """
        Update the enemy's position, making it patrol back and forth horizontally.
        The movement is restricted to ensure the enemy does not move off-screen.
        """
        self.pos['x'] += self.speed * self.direction
        # Reverse direction if the enemy hits the edge of the screen
        if self.pos['x'] <= 0 or self.pos['x'] + self.size >= self.screen_width:
            self.direction *= -1

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the enemy at the current position on the screen.

        Args:
            screen (pygame.Surface): The screen to draw on.
        """
        pygame.draw.rect(screen, self.color, (self.pos['x'], self.pos['y'], self.size, self.size))
