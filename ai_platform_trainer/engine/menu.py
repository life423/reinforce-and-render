# ai_platform_trainer/engine/menu.py
import pygame
import math
from typing import List, Tuple, Optional, Callable, Dict
from ai_platform_trainer.core.color_manager import get_color, lighten, darken

# Spacing scale - based on 8px unit (8, 16, 24, 32, 48, 64)
class Spacing:
    UNIT = 8
    XS = UNIT  # 8px
    SM = UNIT * 2  # 16px
    MD = UNIT * 3  # 24px
    LG = UNIT * 4  # 32px
    XL = UNIT * 6  # 48px
    XXL = UNIT * 8  # 64px

# Typography scale - based on modular scale ratio of 1.25
class Typography:
    # Font sizes follow a modular scale
    BASE = 16
    SCALE = 1.25
    SIZES = {
        "xs": int(BASE * (SCALE ** -1)),  # 13px
        "sm": BASE,  # 16px
        "md": int(BASE * SCALE),  # 20px 
        "lg": int(BASE * (SCALE ** 2)),  # 25px
        "xl": int(BASE * (SCALE ** 3)),  # 31px
        "xxl": int(BASE * (SCALE ** 4)),  # 39px
        "heading": int(BASE * (SCALE ** 5)),  # 49px
        "title": int(BASE * (SCALE ** 6)),  # 61px
    }
    
    @classmethod
    def get_font(cls, size_name: str, bold: bool = False) -> pygame.font.Font:
        """Get a font with the given size and weight."""
        try:
            # First try to load a nicer font if available
            font_name = "Arial"
            size = cls.SIZES[size_name]
            return pygame.font.SysFont(font_name, size, bold=bold)
        except Exception:
            # Fallback to default
            size = cls.SIZES[size_name]
            return pygame.font.SysFont(None, size, bold=bold)


class MenuOption:
    """A class representing a single menu option."""
    
    def __init__(self, text: str, action: Optional[str] = None, 
                 on_select: Optional[Callable] = None, color_key: str = "gray300"):
        """
        Initialize a menu option.
        
        Args:
            text: Text to display for this option
            action: Action identifier to return when option is selected
            on_select: Optional callback to execute when option is selected
            color_key: Color key to use for this option
        """
        self.text = text
        self.action = action or text
        self.on_select = on_select
        self.color_key = color_key
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.animation_scale = 1.0
        self.hover = False
        
        # Animation properties
        self.target_scale = 1.0
        self.current_scale = 1.0
        self.anim_speed = 0.2
    
    def update(self, dt: float, is_active: bool, is_hovered: bool) -> None:
        """Update animation state based on active/hover status."""
        self.hover = is_hovered
        
        # Determine target scale
        if is_active:
            self.target_scale = 1.08  # 8% larger when active
        elif is_hovered:
            self.target_scale = 1.05  # 5% larger when hovered
        else:
            self.target_scale = 1.0
        
        # Smoothly animate toward target scale
        if self.current_scale != self.target_scale:
            direction = 1 if self.target_scale > self.current_scale else -1
            change = min(self.anim_speed * dt, abs(self.target_scale - self.current_scale))
            self.current_scale += direction * change
    
    def render(self, font: pygame.font.Font, is_active: bool) -> Tuple[pygame.Surface, pygame.Rect]:
        """
        Render the menu option with enhanced visual cues when active or hovered.
        
        Args:
            font: Font to use for rendering
            is_active: Whether this option is currently active/selected
            
        Returns:
            Tuple of (rendered surface, rect)
        """
        # Choose appropriate colors based on state
        if is_active:
            # Active options use their specific color at full brightness
            option_num = int(self.color_key.replace('option', '') if 'option' in self.color_key else '1')
            color_key = f"option{option_num}" if option_num <= 8 else "option1"
            text_color = get_color(color_key)
            
            # Use a slightly brighter version for the selection indicator
            indicator_color = lighten(text_color, 0.2)
        elif self.hover:
            # Hovered options use a slightly lighter gray
            text_color = get_color("gray200")
            indicator_color = text_color
        else:
            # Inactive options use their assigned color but slightly muted
            if 'option' in self.color_key:
                text_color = darken(get_color(self.color_key), 0.3)
            else:
                text_color = get_color(self.color_key)
            indicator_color = text_color
        
        # Add selection indicators based on state
        if is_active:
            # Use a subtle indicator for active state (small triangle)
            prefix = "▸ "
            # Bold text for active items
            font_to_use = Typography.get_font("lg", bold=True)
        else:
            prefix = "  "
            font_to_use = font
        
        # Render text with anti-aliasing
        display_text = prefix + self.text
        text_surface = font_to_use.render(display_text, True, text_color)
        text_rect = text_surface.get_rect()
        
        # Apply scaling if needed
        if self.current_scale != 1.0:
            scaled_width = int(text_rect.width * self.current_scale)
            scaled_height = int(text_rect.height * self.current_scale)
            scaled_surface = pygame.Surface((scaled_width, scaled_height), pygame.SRCALPHA)
            
            # Scale the text surface smoothly
            scaled_text = pygame.transform.smoothscale(text_surface, (scaled_width, scaled_height))
            scaled_surface.blit(scaled_text, (0, 0))
            
            return scaled_surface, scaled_surface.get_rect()
        else:
            return text_surface, text_rect
    
    def select(self) -> str:
        """Handle selection of this option."""
        if self.on_select:
            self.on_select()
        return self.action


class MenuSection:
    """A section of the menu that manages a collection of options."""
    
    def __init__(self, title: Optional[str] = None, 
                 options: Optional[List[MenuOption]] = None):
        """
        Initialize a menu section.
        
        Args:
            title: Optional title for this section
            options: List of MenuOption objects
        """
        self.title = title
        self.options = options or []
        self.active_index = 0
        self.hover_index = -1  # -1 means no hover
    
    def add_option(self, option: MenuOption) -> None:
        """Add an option to this section."""
        self.options.append(option)
    
    def move_selection(self, direction: int) -> None:
        """
        Move the selection up or down.
        
        Args:
            direction: -1 for up, 1 for down
        """
        if not self.options:
            return
        self.active_index = (self.active_index + direction) % len(self.options)
        # Reset hover when using keyboard navigation
        self.hover_index = -1
    
    def get_active_option(self) -> Optional[MenuOption]:
        """Get the currently active option."""
        if not self.options:
            return None
        return self.options[self.active_index]
    
    def select_active(self) -> Optional[str]:
        """Select the currently active option."""
        active = self.get_active_option()
        return active.select() if active else None
    
    def update_hover(self, mouse_pos: Tuple[int, int]) -> None:
        """Update which option is being hovered based on mouse position."""
        self.hover_index = -1
        for i, option in enumerate(self.options):
            if option.rect and option.rect.collidepoint(mouse_pos):
                self.hover_index = i
                break
    
    def handle_mouse_click(self, pos: Tuple[int, int]) -> Optional[str]:
        """
        Check if any option was clicked and select it.
        
        Args:
            pos: Mouse position (x, y)
            
        Returns:
            Action string if an option was clicked, None otherwise
        """
        for i, option in enumerate(self.options):
            if option.rect and option.rect.collidepoint(pos):
                self.active_index = i
                return self.select_active()
        return None
    
    def update(self, dt: float, mouse_pos: Tuple[int, int]) -> None:
        """Update all options in this section."""
        self.update_hover(mouse_pos)
        
        for i, option in enumerate(self.options):
            is_active = (i == self.active_index)
            is_hovered = (i == self.hover_index)
            option.update(dt, is_active, is_hovered)

    def render(self, screen: pygame.Surface, font: pygame.font.Font, 
               section_pos: Tuple[int, int]) -> None:
        """
        Render the entire section with title and options.
        
        Args:
            screen: Surface to render onto
            font: Font to use for rendering options
            section_pos: Position (x, y) for the section title
        """
        x, y = section_pos
        
        # Render title if present with larger font
        if self.title:
            title_font = Typography.get_font("heading", bold=True)
            title_surf = title_font.render(self.title, True, get_color("primary"))
            title_rect = title_surf.get_rect(center=(x, y))
            screen.blit(title_surf, title_rect)
            y += Spacing.XXL  # Large spacing after title
        
        # Assign unique colors to each option for variety
        option_colors = [
            "option1", "option2", "option3", "option4", 
            "option5", "option6", "option7", "option8"
        ]
        
        # Render options
        for i, option in enumerate(self.options):
            # Set color for this option if not already set
            if option.color_key == "gray300":
                option.color_key = option_colors[i % len(option_colors)]
                
            is_active = (i == self.active_index)
            surf, rect = option.render(font, is_active)
            
            # Position with proper spacing
            spacing = Spacing.LG + int(i * 1.5)  # Slight increase for each item
            rect.center = (x, y + i * spacing)
            option.rect = rect
            
            # Draw subtle indicator for active option
            if is_active:
                # Draw an animated underline
                time_factor = pygame.time.get_ticks() % 2000 / 2000  # 0.0 to 1.0 cycling every 2 seconds
                line_width = int(rect.width * (0.8 + 0.2 * math.sin(time_factor * math.pi * 2)))
                
                underline_rect = pygame.Rect(
                    rect.centerx - line_width // 2,
                    rect.bottom + 4,
                    line_width,
                    2
                )
                
                # Get the option's color and make it semi-transparent
                color = get_color(option.color_key)
                alpha_color = (*color, 150)  # 150/255 opacity
                
                # Create a surface for the line with alpha channel
                line_surf = pygame.Surface((underline_rect.width, underline_rect.height), pygame.SRCALPHA)
                line_surf.fill(alpha_color)
                
                # Draw the line
                screen.blit(line_surf, underline_rect)
            
            # Blit the option surface
            screen.blit(surf, rect)


class MainMenu:
    """Main menu controller class."""
    
    def __init__(self, screen: pygame.Surface):
        """
        Initialize the menu.
        
        Args:
            screen: Pygame surface to render onto
        """
        self.screen = screen
        self.width, self.height = screen.get_size()
        self.done = False
        self.help_on = False
        self.full = False
        self.action = None
        self.last_time = pygame.time.get_ticks()
        self.mouse_pos = (0, 0)
        
        # Initialize fonts using the typography system
        self._init_fonts()
        
        # Create main menu section
        self.main_section = self._setup_main_section()
    
    def _init_fonts(self):
        """Initialize fonts with proper typographic scale."""
        self.title_font = Typography.get_font("title", bold=True)
        self.heading_font = Typography.get_font("heading", bold=True)
        self.option_font = Typography.get_font("lg")
        self.small_font = Typography.get_font("md")
        self.tiny_font = Typography.get_font("sm")
    
    def _setup_main_section(self) -> MenuSection:
        """Setup the main menu options."""
        section = MenuSection(title=None)  # No title, it's drawn separately
        
        # Add required menu options
        section.add_option(MenuOption("Play RL Game", color_key="option1"))
        section.add_option(MenuOption("Train RL Model (live)", color_key="option2"))
        section.add_option(MenuOption("Train RL Model (headless)", color_key="option3"))
        section.add_option(MenuOption("Train Supervised Model (live)", color_key="option4"))
        section.add_option(MenuOption("Train Supervised Model (headless)", color_key="option5"))
        section.add_option(MenuOption("Help / Controls", on_select=self._toggle_help, color_key="option6"))
        section.add_option(
            MenuOption("Toggle Fullscreen", on_select=self._toggle_fullscreen, color_key="option7")
        )
        section.add_option(MenuOption("Quit", on_select=self._quit, color_key="option8"))
        
        return section
    
    def _toggle_help(self) -> None:
        """Toggle the help screen."""
        self.help_on = not self.help_on
    
    def _toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        self.full = not self.full
        flags = (pygame.FULLSCREEN | pygame.SCALED | pygame.DOUBLEBUF 
                if self.full else pygame.SCALED | pygame.DOUBLEBUF)
        pygame.display.set_mode(
            (0, 0) if self.full else (800, 600), 
            flags, 
            vsync=1
        )
        self.screen = pygame.display.get_surface()
        self.width, self.height = self.screen.get_size()
    
    def _quit(self) -> None:
        """Handle quit action."""
        self.done = True
    
    def _process_selection(self, action: Optional[str]) -> None:
        """
        Process a menu selection.
        
        Args:
            action: The action string from the selected option
        """
        if not action:
            return
            
        if action == "Quit":
            self.done = True
        elif action not in ("Toggle Fullscreen", "Help / Controls"):
            self.action = action
            self.done = True
    
    def update(self) -> None:
        """Update menu state and animations."""
        # Calculate delta time
        current_time = pygame.time.get_ticks()
        dt = (current_time - self.last_time) / 1000.0  # Convert to seconds
        self.last_time = current_time
        
        # Get current mouse position
        self.mouse_pos = pygame.mouse.get_pos()
        
        # Update menu section
        if not self.help_on:
            self.main_section.update(dt, self.mouse_pos)
    
    def run(self) -> Optional[str]:
        """
        Run the menu loop.
        
        Returns:
            Selected action string or None if quitting
        """
        clock = pygame.time.Clock()
        self.last_time = pygame.time.get_ticks()
        
        while not self.done:
            # Handle events
            self._handle_events()
            
            # Update menu state
            self.update()
            
            # Draw everything
            self._draw()
            
            # Update the display and maintain framerate
            pygame.display.flip()
            clock.tick(60)
        
        return self.action
    
    def _handle_events(self) -> None:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                return
            
            if self.help_on:
                self._handle_help_events(event)
                continue
            
            if event.type == pygame.KEYDOWN:
                self._handle_keydown(event)
                
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_mouse_click(event.pos)
                
            # Track mouse motion for hover effects
            if event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
    
    def _handle_help_events(self, event) -> None:
        """Handle events when help screen is displayed."""
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_h, pygame.K_q):
                self.help_on = False
    
    def _handle_keydown(self, event) -> None:
        """Handle key press events."""
        # Navigation
        if event.key in (pygame.K_UP, pygame.K_w):
            self.main_section.move_selection(-1)
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self.main_section.move_selection(1)
        
        # Selection
        elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self._process_selection(self.main_section.select_active())
        
        # Help toggle
        elif event.key == pygame.K_h:
            self._toggle_help()
        
        # Exit
        elif event.key in (pygame.K_ESCAPE, pygame.K_q):
            self.done = True
    
    def _handle_mouse_click(self, pos) -> None:
        """Handle mouse click events."""
        selection = self.main_section.handle_mouse_click(pos)
        if selection:
            self._process_selection(selection)
    
    def _draw(self) -> None:
        """Draw the menu content."""
        # Clear the screen with background color
        self.screen.fill(get_color("bg"))
        
        if self.help_on:
            self._draw_help()
        else:
            self._draw_main_menu()
    
    def _draw_main_menu(self) -> None:
        """Draw the main menu screen with improved typography."""
        # Draw title with the larger title font
        title_surface = self.title_font.render(
            "Reinforce-and-Render", 
            True, 
            get_color("primary")
        )
        title_rect = title_surface.get_rect(center=(self.width//2, self.height//4))
        
        # Draw a gradient underline under the title
        underline_width = title_rect.width * 0.8
        underline_height = 4
        underline_rect = pygame.Rect(
            title_rect.centerx - underline_width // 2,
            title_rect.bottom + Spacing.XS,
            underline_width,
            underline_height
        )
        
        # Create gradient colors
        gradient_surface = pygame.Surface((underline_width, underline_height))
        
        # Create a horizontal gradient from primary to accent color
        primary_color = get_color("primary")
        accent_color = get_color("accent")
        
        for x in range(int(underline_width)):
            # Calculate blend factor (0 to 1)
            blend = x / underline_width
            # Interpolate between primary and accent color
            color = [int(primary_color[i] * (1 - blend) + accent_color[i] * blend) for i in range(3)]
            pygame.draw.line(gradient_surface, color, (x, 0), (x, underline_height))
        
        # Blit the title and gradient
        self.screen.blit(title_surface, title_rect)
        self.screen.blit(gradient_surface, underline_rect)
        
        # Draw menu options with proper spacing
        self.main_section.render(
            self.screen,
            self.option_font,
            (self.width//2, title_rect.bottom + Spacing.XXL)
        )
        
        # Draw footer with smaller font and proper positioning
        footer_text = "↑/↓ to select · ↵ to confirm · H for help"
        footer_surface = self.small_font.render(
            footer_text, 
            True, 
            get_color("gray500")
        )
        footer_rect = footer_surface.get_rect(center=(self.width//2, self.height - Spacing.XL))
        self.screen.blit(footer_surface, footer_rect)
    
    def _draw_help(self) -> None:
        """Draw the help screen with improved typography."""
        # Title for help screen
        help_title = self.heading_font.render(
            "Help & Controls", 
            True, 
            get_color("primary")
        )
        help_title_rect = help_title.get_rect(center=(self.width//2, Spacing.XXL))
        self.screen.blit(help_title, help_title_rect)
        
        # Draw decorative header bar
        header_bar_rect = pygame.Rect(
            self.width//4, 
            help_title_rect.bottom + Spacing.SM,
            self.width//2,
            2
        )
        pygame.draw.rect(self.screen, get_color("primary"), header_bar_rect)
        
        # Help content with better spacing and colors
        help_sections = [
            {
                "title": "Player Controls",
                "color": "option1",
                "items": [
                    "WASD / Arrow Keys - Move",
                    "Esc / Q - Quit"
                ]
            },
            {
                "title": "Menu Navigation",
                "color": "option2",
                "items": [
                    "↑/↓ or W/S - Navigate options",
                    "Enter / Space - Select option",
                    "H - Toggle this help screen"
                ]
            },
            {
                "title": "Training Modes",
                "color": "option4",
                "items": [
                    "Live - Opens window to observe rewards",
                    "Headless - Runs in background (faster)"
                ]
            }
        ]
        
        # Draw each section with proper positioning and spacing
        y = help_title_rect.bottom + Spacing.XL
        
        for section in help_sections:
            # Draw section title
            title_color = get_color(section["color"])
            title_surf = self.option_font.render(section["title"] + ":", True, title_color)
            title_rect = title_surf.get_rect(midleft=(self.width//4, y))
            self.screen.blit(title_surf, title_rect)
            
            # Draw section items
            item_y = title_rect.bottom + Spacing.SM
            for item in section["items"]:
                item_surf = self.small_font.render(item, True, get_color("gray300"))
                item_rect = item_surf.get_rect(midleft=(self.width//4 + Spacing.MD, item_y))
                self.screen.blit(item_surf, item_rect)
                item_y += Spacing.MD
            
            # Add spacing between sections
            y = item_y + Spacing.LG
        
        # Draw footer with subtle animation
        time_factor = (pygame.time.get_ticks() % 3000) / 3000  # cycles between 0 and 1 every 3 seconds
        alpha = int(200 + 55 * math.sin(time_factor * math.pi * 2))  # Oscillate between 145 and 255
        
        esc_surf = self.small_font.render("Press Esc to return", True, get_color("accent"))
        esc_rect = esc_surf.get_rect(center=(self.width//2, self.height - Spacing.XL))
        
        # Create a copy with changing alpha for pulsing effect
        esc_surf_copy = esc_surf.copy()
        esc_surf_copy.set_alpha(alpha)
        self.screen.blit(esc_surf_copy, esc_rect)
