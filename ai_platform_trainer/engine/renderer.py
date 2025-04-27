import pygame

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
        self.screen.fill(color)

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
