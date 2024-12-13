import pygame
from ai_platform_trainer.entities.player import Player


class PlayerPlay(Player):
    def handle_input(self):
        """Handle player keyboard input for movement in play mode.
        Returns False if ESC is pressed (to stop the game), True otherwise."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.position["x"] -= self.step
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.position["x"] += self.step
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.position["y"] -= self.step
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.position["y"] += self.step
        if keys[pygame.K_ESCAPE]:
            return False  # Signal the game to stop

        # After applying movement, clamp the position.
        self.clamp_position()
        return True

    def update(self, enemy_x, enemy_y):
        # In play mode, movement is primarily handled by handle_input().
        pass
