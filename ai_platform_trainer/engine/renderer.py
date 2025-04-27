import pygame
from ai_platform_trainer.core.color_manager import get_color

class Renderer:
    def __init__(self, screen: pygame.Surface):
        """
        Receives the main display surface.
        """
        self.screen = screen

    def clear(self, color=(0,0,0)) -> None:
        """
        Optionally clear the screen to a solid color.
        """
        # Accept a role name or raw RGB tuple
        fill_color = get_color(color) if isinstance(color, str) else color
        self.screen.fill(fill_color)

    def draw(self, drawable) -> None:
        """
        Draw any object that implements .draw(surface).
        """
        drawable.draw(self.screen)

    def present(self):
        """
        Flip the display buffers.
        """
        pygame.display.flip()
