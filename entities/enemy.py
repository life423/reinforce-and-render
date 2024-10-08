import pygame
import random


class Enemy:
    def __init__(
        self, start_x: int, start_y: int, screen_width: int, screen_height: int
    ) -> None:
        """
        Initialize the enemy with a starting position and screen constraints.
        """
        self.pos = {"x": start_x, "y": start_y}
        self.size = 100  # Size of the enemy block
        self.color = (255, 69, 0)  # Orange-Red color for the enemy
        self.speed = max(1, screen_width // 500)  # Speed of enemy movement
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.movement_counter = 0
        self.current_direction = random.choice(
            [(-self.speed, 0), (self.speed, 0), (0, -self.speed), (0, self.speed)]
        )

    def update_position(self, player_pos: dict) -> None:
        """
        Update the position of the enemy based on the player's position.
        The movement is varied to create more dynamic behavior.

        Args:
            player_pos (dict): The current position of the player.
        """
        # Change direction every 20 frames to make movement more varied
        if self.movement_counter % 20 == 0:
            self.current_direction = random.choice(
                [(-self.speed, 0), (self.speed, 0), (0, -self.speed), (0, self.speed)]
            )

        dx, dy = self.current_direction

        new_x = self.pos["x"] + dx
        new_y = self.pos["y"] + dy

        # Ensure the enemy does not move off-screen
        self.pos["x"] = max(0, min(self.screen_width - self.size, new_x))
        self.pos["y"] = max(0, min(self.screen_height - self.size, new_y))

        # Increment movement counter
        self.movement_counter += 1

    def update_position_randomly(self) -> None:
        """
        Update the position of the enemy randomly.
        """
        # Change direction every 20 frames to make movement more varied
        if self.movement_counter % 20 == 0:
            self.current_direction = random.choice(
                [(-self.speed, 0), (self.speed, 0), (0, -self.speed), (0, self.speed)]
            )

        dx, dy = self.current_direction

        new_x = self.pos["x"] + dx
        new_y = self.pos["y"] + dy

        # Ensure the enemy does not move off-screen
        self.pos["x"] = max(0, min(self.screen_width - self.size, new_x))
        self.pos["y"] = max(0, min(self.screen_height - self.size, new_y))

        # Increment movement counter
        self.movement_counter += 1

    def draw(self, screen) -> None:
        """
        Draw the enemy on the screen.
        """
        pygame.draw.rect(
            screen, self.color, (self.pos["x"], self.pos["y"], self.size, self.size)
        )
