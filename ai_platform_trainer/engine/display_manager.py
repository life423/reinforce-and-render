import pygame

class DisplayManager:
    def __init__(self, width: int, height: int, title: str = "AI Trainer"):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

    def get_screen(self) -> pygame.Surface:
        return self.screen

    def tick(self, fps: int) -> None:
        self.clock.tick(fps)

    def update(self) -> None:
        pygame.display.flip()

    def quit(self) -> None:
        pygame.quit()
