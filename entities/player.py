import pygame


class Player:
    def __init__(self, screen_width, screen_height):
        self.position = {"x": screen_width // 2, "y": screen_height // 2}
        self.size = 50
        self.color = (0, 102, 204)
        self.step = 5
        self.screen_width = screen_width
        self.screen_height = screen_height

    def reset(self):
        # Reset player position to the center of the screen
        self.position = {"x": self.screen_width //
                         2, "y": self.screen_height // 2}

    def update(self):
        keys = pygame.key.get_pressed()
        # Check for movement keys and update position accordingly
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.position["x"] -= self.step
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.position["x"] += self.step
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.position["y"] -= self.step
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.position["y"] += self.step

        # Ensure the player doesn't move off-screen
        self.position["x"] = max(
            0, min(self.screen_width - self.size, self.position["x"]))
        self.position["y"] = max(
            0, min(self.screen_height - self.size, self.position["y"]))

    def draw(self, screen):
        pygame.draw.rect(
            screen, self.color, (self.position["x"], self.position["y"], self.size, self.size))

    def get_position(self):
        """
        Get the current position of the player.

        Returns:
            dict: The player's current position as {"x": ..., "y": ...}
        """
        return self.position
