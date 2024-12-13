import pygame
class Enemy:
    def __init__(self, screen_width, screen_height):
        self.start_x = screen_width // 2
        self.start_y = screen_height // 2
        self.pos = {"x": self.start_x, "y": self.start_y}
        self.size = 50
        self.color = (255, 165, 0)
        self.speed = max(2, screen_width // 400)
        self.screen_width = screen_width
        self.screen_height = screen_height

    def reset(self):
        self.pos = {"x": self.start_x, "y": self.start_y}
    
    def update_movement(self, player_x, player_y, player_speed):
    # Step 2: Compute direction vector towards the player
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]

        # Step 3: Normalize the direction vector
        dist = (dx**2 + dy**2)**0.5
        if dist > 0:
            dx /= dist
            dy /= dist

        # Step 4: Set enemy speed to 70% of player_speed
        enemy_speed = player_speed * 0.7

        # Move enemy towards the player
        self.pos["x"] += dx * enemy_speed
        self.pos["y"] += dy * enemy_speed

        # Step 5: Clamp position to screen bounds
        self.pos["x"] = max(0, min(self.screen_width - self.size, self.pos["x"]))
        self.pos["y"] = max(0, min(self.screen_height - self.size, self.pos["y"]))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color,
                         (self.pos["x"], self.pos["y"], self.size, self.size))
