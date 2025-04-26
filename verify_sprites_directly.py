#!/usr/bin/env python
"""
Script to directly verify all game sprites by rendering them to a screenshot.

This script initializes pygame and the sprite manager, then renders all
sprite types to a grid with labels for easy verification.
"""
import os
import pygame

from ai_platform_trainer.utils.sprite_manager import SpriteManager


def main():
    """Create a visual verification of all game sprites."""
    print("Starting direct sprite verification...")
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Sprite Verification")
    
    # Create a sprite manager
    sprite_manager = SpriteManager()
    
    # Set up grid coordinates for sprite display
    sprite_size = (64, 64)
    sprites_to_verify = [
        "player", "enemy", "missile", "wall", "rock", "obstacle",
        "explosion_0", "explosion_1", "explosion_2", "explosion_3"
    ]
    
    # Fill screen with dark background
    screen.fill((20, 20, 30))
    
    # Create a font for labels
    font = pygame.font.SysFont('Arial', 16)
    
    # Render each sprite in a grid with labels
    cols = 3
    for i, sprite_name in enumerate(sprites_to_verify):
        # Calculate grid position
        row = i // cols
        col = i % cols
        x = col * 200 + 100
        y = row * 150 + 100
        
        # Render sprite
        sprite = sprite_manager.load_sprite(sprite_name, sprite_size)
        screen.blit(sprite, (x - sprite_size[0]//2, y - sprite_size[1]//2))
        
        # Check if this is a placeholder sprite
        is_placeholder = sprite_manager.is_placeholder(sprite)
        status = "✓ Real Sprite" if not is_placeholder else "✗ Placeholder"
        
        # Render label
        name_text = font.render(sprite_name, True, (255, 255, 255))
        status_text = font.render(status, True, 
                                 (100, 255, 100) if not is_placeholder else (255, 100, 100))
        
        screen.blit(name_text, (x - name_text.get_width()//2, y + sprite_size[1]//2 + 5))
        screen.blit(status_text, (x - status_text.get_width()//2, y + sprite_size[1]//2 + 25))
    
    # Add a title
    title_font = pygame.font.SysFont('Arial', 24, bold=True)
    title = title_font.render("Game Sprite Verification", True, (200, 200, 255))
    screen.blit(title, (400 - title.get_width()//2, 30))
    
    # Add instructions
    instructions = font.render("All sprites should display properly (not placeholders)", 
                             True, (200, 200, 200))
    screen.blit(instructions, (400 - instructions.get_width()//2, 60))
    
    # Take a screenshot
    pygame.display.flip()
    screenshot_path = "sprites_verification.png"
    pygame.image.save(screen, screenshot_path)
    print(f"Verification screenshot saved to {os.path.abspath(screenshot_path)}")
    
    # Clean exit
    pygame.quit()
    print("Verification complete. Check the screenshot to confirm all sprites display correctly.")


if __name__ == "__main__":
    main()