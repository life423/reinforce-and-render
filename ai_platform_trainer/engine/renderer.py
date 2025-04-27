import pygame
import logging
from typing import Union, Tuple, Optional, Any

from ai_platform_trainer.core.color_manager import (
    get_color, set_palette, get_current_palette,
    lighten, darken, rgb_to_hex, ColorType
)

class Renderer:
    def __init__(self, screen: pygame.Surface):
        """
        Initialize the renderer with the main display surface.
        
        Args:
            screen: The pygame surface to render to
        """
        self.screen = screen
        self.logger = logging.getLogger(__name__)
        self._debug_mode = False
        self._current_theme = get_current_palette()
        
    def clear(self, color: Union[str, Tuple[int, int, int]] = "background") -> None:
        """
        Clear the screen to a solid color.
        
        Args:
            color: Color role name from color_manager or RGB tuple
                  Default is "background" from current palette
        """
        # Accept a role name or raw RGB tuple
        try:
            fill_color = get_color(color) if isinstance(color, str) else color
            self.screen.fill(fill_color)
        except ValueError as e:
            self.logger.warning(f"Invalid color in clear(): {e}. Using black.")
            self.screen.fill((0, 0, 0))

    def draw(self, drawable: Any) -> None:
        """
        Draw any object that implements .draw(surface).
        
        Args:
            drawable: An object with a draw(surface) method
        """
        try:
            drawable.draw(self.screen)
        except Exception as e:
            self.logger.error(f"Error drawing object {drawable}: {e}")
            
    def present(self) -> None:
        """Flip the display buffers to show the rendered frame."""
        pygame.display.flip()
        
    def set_theme(self, palette_name: str) -> None:
        """
        Change the current color theme for the entire application.
        
        Args:
            palette_name: The name of a predefined color palette 
                         ("default", "monochrome", "high_contrast")
        """
        try:
            set_palette(palette_name)
            self._current_theme = palette_name
            self.logger.info(f"Changed color theme to: {palette_name}")
        except ValueError as e:
            self.logger.error(f"Failed to change theme: {e}")
            
    def get_theme(self) -> str:
        """Return the current active theme/palette name."""
        return self._current_theme
    
    def draw_text(
        self, 
        text: str, 
        position: Tuple[int, int],
        font_size: int = 20,
        color: Union[str, Tuple[int, int, int]] = "text",
        centered: bool = False,
        background: Optional[Union[str, ColorType]] = None
    ) -> pygame.Rect:
        """
        Draw text on the screen with advanced options.
        
        Args:
            text: The text to render
            position: (x, y) coordinates for the text
            font_size: Size of the font in points
            color: Text color (role name or RGB tuple)
            centered: If True, position is the center of the text, not top-left
            background: Optional background color (role name or RGB)
            
        Returns:
            The rectangle containing the rendered text
        """
        try:
            # Get the actual RGB values if color roles are provided
            text_color = get_color(color) if isinstance(color, str) else color
            bg_color = get_color(background) if isinstance(background, str) and background else background
            
            # Create font and render text
            font = pygame.font.SysFont("Arial", font_size)
            text_surface = font.render(text, True, text_color, bg_color)
            
            # Calculate position for centered text
            text_rect = text_surface.get_rect()
            if centered:
                text_rect.center = position
            else:
                text_rect.topleft = position
                
            # Draw to screen
            self.screen.blit(text_surface, text_rect)
            return text_rect
            
        except Exception as e:
            self.logger.error(f"Failed to render text: {e}")
            return pygame.Rect(0, 0, 0, 0)
    
    def draw_debug_info(self, fps: float, entity_count: int) -> None:
        """
        Draw debug information overlay if debug mode is enabled.
        
        Args:
            fps: Current frames per second
            entity_count: Number of entities in the scene
        """
        if not self._debug_mode:
            return
            
        # Draw semi-transparent panel
        debug_surface = pygame.Surface((200, 80), pygame.SRCALPHA)
        debug_surface.fill((*get_color("panel"), 180))  # Using panel color with alpha
        self.screen.blit(debug_surface, (10, 10))
        
        # Draw debug text
        self.draw_text(f"FPS: {fps:.1f}", (20, 20), font_size=14, color="text")
        self.draw_text(f"Entities: {entity_count}", (20, 40), font_size=14, color="text")
        palette = self.get_theme()
        self.draw_text(f"Theme: {palette}", (20, 60), font_size=14, color="text")
        
    def toggle_debug_mode(self) -> bool:
        """
        Toggle debug information overlay.
        
        Returns:
            The new debug mode state
        """
        self._debug_mode = not self._debug_mode
        return self._debug_mode
    
    def draw_ui_element(
        self,
        rect: pygame.Rect,
        color: Union[str, ColorType] = "panel",
        border_radius: int = 0,
        border_color: Optional[Union[str, ColorType]] = None,
        border_width: int = 0
    ) -> None:
        """
        Draw a UI element with optional rounded corners and border.
        
        Args:
            rect: The rectangle defining the element bounds
            color: The fill color (role name or RGB)
            border_radius: Radius for rounded corners (0 for sharp corners)
            border_color: Optional border color (role name or RGB)
            border_width: Border width in pixels
        """
        # Get actual colors
        fill_color = get_color(color) if isinstance(color, str) else color
        
        # Draw the main element
        if border_radius > 0:
            pygame.draw.rect(self.screen, fill_color, rect, border_radius=border_radius)
        else:
            pygame.draw.rect(self.screen, fill_color, rect)
            
        # Draw border if specified
        if border_color and border_width > 0:
            border_rgb = get_color(border_color) if isinstance(border_color, str) else border_color
            if border_radius > 0:
                pygame.draw.rect(self.screen, border_rgb, rect, width=border_width, border_radius=border_radius)
            else:
                pygame.draw.rect(self.screen, border_rgb, rect, width=border_width)
