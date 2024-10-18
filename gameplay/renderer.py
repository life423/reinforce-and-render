# gameplay/renderer.py
import pygame


def initialize_screen(screen_title: str) -> pygame.Surface:
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption(screen_title)
    return screen

