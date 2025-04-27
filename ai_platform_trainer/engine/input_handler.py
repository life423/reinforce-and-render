import pygame

class InputHandler:
    def get_actions(self) -> dict:
        actions: dict = {}
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions["quit"] = True
        return actions
