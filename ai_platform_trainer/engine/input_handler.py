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

        # 1) Catch quit events (window close) and ESC keydown
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                actions['quit'] = True
            elif evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE:
                actions['quit'] = True

        # 2) Poll keys for smooth continuous movement and ESC hold
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]    or keys[pygame.K_UP]:
            actions['up'] = True
        if keys[pygame.K_s]    or keys[pygame.K_DOWN]:
            actions['down'] = True
        if keys[pygame.K_a]    or keys[pygame.K_LEFT]:
            actions['left'] = True
        if keys[pygame.K_d]    or keys[pygame.K_RIGHT]:
            actions['right'] = True
        if keys[pygame.K_ESCAPE]:
            actions['quit'] = True

        return actions
