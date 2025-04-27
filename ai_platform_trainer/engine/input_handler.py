import pygame

class InputHandler:
    def __init__(self):
        pygame.init()

    def get_actions(self) -> dict[str, bool]:
        """
        Polls Pygame events and returns a dict of current actions:
          'up', 'down', 'left', 'right', 'quit'
        
        Uses continuous keyboard polling for smooth movement.
        """
        actions = {'up': False, 'down': False, 'left': False, 'right': False, 'quit': False}

        # Flush events queue and check for QUIT and ESC
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                actions['quit'] = True
            
        # Poll keys for continuous movement each frame
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            actions['up'] = True
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            actions['down'] = True
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            actions['left'] = True
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            actions['right'] = True
        
        # ESC key press or hold -> quit
        if keys[pygame.K_ESCAPE]:
            actions['quit'] = True

        return actions
