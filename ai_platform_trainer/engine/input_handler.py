# ai_platform_trainer/engine/input_handler.py
import pygame

class InputHandler:
    def __init__(self):
        pygame.init()

    def get_actions(self):
        """
        Returns a dict:
          'move_x': -1, 0, or +1
          'move_y': -1, 0, or +1
          'quit': True if player requested exit
        """
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                return {'move_x': 0, 'move_y': 0, 'quit': True}

        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
        dy = (keys[pygame.K_DOWN]  - keys[pygame.K_UP])
        quit_ = keys[pygame.K_ESCAPE]
        return {'move_x': dx, 'move_y': dy, 'quit': quit_}
