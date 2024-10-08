import pygame

class Player:
    def __init__(self, screen_width: int, screen_height: int, config):
        self.pos = {'x': screen_width // 2, 'y': screen_height // 2}
        self.size = config.PLAYER_SIZE
        self.color = config.PLAYER_COLOR
        self.step = config.PLAYER_STEP

    def move(self, dx: int, dy: int):
        new_x = self.pos['x'] + dx
        new_y = self.pos['y'] + dy
        self.pos['x'] = max(0, min(screen_width - self.size, new_x))
        self.pos['y'] = max(0, min(screen_height - self.size, new_y))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.pos['x'], self.pos['y'], self.size, self.size))