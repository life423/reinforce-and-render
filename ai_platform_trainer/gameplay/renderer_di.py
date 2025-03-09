# file: ai_platform_trainer/gameplay/renderer_di.py
"""
Concrete renderer implementation that implements the IRenderer interface.
"""
import pygame
from ai_platform_trainer.core.interfaces import IRenderer


class Renderer(IRenderer):
    """
    Renderer class for drawing game elements to the screen.
    Implements the IRenderer interface.
    """
    
    def __init__(self, screen):
        """
        Initialize the renderer.
        
        Args:
            screen: The pygame screen to render to
        """
        self.screen = screen
    
    def render(self, menu=None, player=None, enemy=None, menu_active=False):
        """
        Render game elements to the screen.
        
        Args:
            menu: The menu to render (if any)
            player: The player to render (if any)
            enemy: The enemy to render (if any)
            menu_active: Whether the menu is active
        """
        # Clear the screen
        self.screen.fill((0, 0, 0))
        
        if menu_active and menu:
            # If menu is active, let the menu handle rendering
            menu.draw(self.screen)
        else:
            # Render game elements
            if player:
                # Render the player
                pygame.draw.rect(
                    self.screen,
                    player.color,
                    (
                        player.position["x"],
                        player.position["y"],
                        player.size,
                        player.size,
                    ),
                )
                
                # Render player missiles
                for missile in player.missiles:
                    pygame.draw.rect(
                        self.screen,
                        missile.color,
                        (
                            missile.position["x"],
                            missile.position["y"],
                            missile.size,
                            missile.size,
                        ),
                    )
            
            if enemy and not enemy.hidden:
                # Render the enemy with current alpha (for fade-in effect)
                s = pygame.Surface((enemy.size, enemy.size), pygame.SRCALPHA)
                pygame.draw.rect(
                    s,
                    enemy.color + (enemy.alpha,),  # RGBA with current alpha
                    (0, 0, enemy.size, enemy.size),
                )
                self.screen.blit(s, (enemy.pos["x"], enemy.pos["y"]))
