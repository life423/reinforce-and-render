import pygame


class Player:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.step = 5
        self.color = (0, 102, 204)  # A shade of blue
        self.position = {"x": self.screen_width // 2, "y": self.screen_height // 2}

    def reset(self):
        self.position = {"x": self.screen_width // 2, "y": self.screen_height // 2}



    def update(self, enemy_x, enemy_y):
        # Base class does nothing; play/training modes override if needed.
        pass

    def draw(self, screen):
        # Draw the player as a rectangle
        pygame.draw.rect(
            screen,
            self.color,
            (self.position["x"], self.position["y"], self.size, self.size),
        )
