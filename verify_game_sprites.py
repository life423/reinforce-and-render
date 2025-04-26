#!/usr/bin/env python
"""
Script to verify that sprites load correctly during gameplay.

This script launches the game, lets it run for a few seconds to ensure
everything renders, and takes a screenshot to verify sprite loading.
"""
import os

import pygame

from ai_platform_trainer.engine.core.game import Game


def main():
    """Run the game for a few seconds and take a screenshot."""
    print("Starting game to verify sprites...")
    game = Game()
    
    # Skip menu and directly start in play mode
    game.start_game("play")
    game.menu_active = False
    
    # Run for a few frames to ensure everything loads
    for _ in range(10):
        current_time = pygame.time.get_ticks()
        game.handle_events()
        game.update(current_time)
        game.renderer.render(game.menu, game.player, game.enemy, game.menu_active)
        pygame.display.flip()
        game.clock.tick(30)
    
    # Take a screenshot
    screenshot_path = "game_sprites_screenshot.png"
    pygame.image.save(game.screen, screenshot_path)
    print(f"Screenshot saved to {os.path.abspath(screenshot_path)}")
    
    # Clean exit
    pygame.quit()
    print("Verification complete. Check the screenshot to confirm sprites display correctly.")



if __name__ == "__main__":
    main()