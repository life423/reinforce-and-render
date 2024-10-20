import pygame


class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.BACKGROUND_COLOR = (135, 206, 235)  # Light blue

    def render(self, menu, player, enemy, menu_active, screen):
        self.screen.fill(self.BACKGROUND_COLOR)
        if menu_active:
            menu.draw(self.screen)
        else:
            player.draw(self.screen)
            enemy.draw(self.screen)
        pygame.display.flip()
