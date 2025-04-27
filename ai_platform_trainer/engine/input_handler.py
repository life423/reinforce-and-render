import pygame

class InputHandler:
    def __init__(self):
        pygame.init()

    def get_actions(self) -> dict[str, bool]:
        """
        Polls Pygame events and returns a dict of current actions:
          'up', 'down', 'left', 'right', 'quit'
        """
        actions = {'up': False, 'down': False, 'left': False, 'right': False, 'quit': False}
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                actions['quit'] = True
            elif evt.type == pygame.KEYDOWN:
                match evt.key:
                    case pygame.K_ESCAPE:
                        actions['quit'] = True
                    case pygame.K_w | pygame.K_UP:
                        actions['up'] = True
                    case pygame.K_s | pygame.K_DOWN:
                        actions['down'] = True
                    case pygame.K_a | pygame.K_LEFT:
                        actions['left'] = True
                    case pygame.K_d | pygame.K_RIGHT:
                        actions['right'] = True
        return actions
