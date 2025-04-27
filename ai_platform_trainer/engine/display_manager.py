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

    def clear(self, color_role: str = "primary") -> None:
        # Uses ColorManager if needed; default black fallback
        from ai_platform_trainer.core.color_manager import get_color
        fill_color = get_color(color_role) if isinstance(color_role, str) else color_role
        self.screen.fill(fill_color)

    def update(self) -> None:
        pygame.display.flip()

    def quit(self) -> None:
        pygame.quit()