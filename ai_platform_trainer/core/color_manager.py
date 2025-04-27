# ai_platform_trainer/core/color_manager.py

from typing import Tuple, Dict, Union, Optional

ColorType = Tuple[int, int, int]

# Expanded color palette following color theory principles
_ROLE_COLORS: Dict[str, ColorType] = {
    # Core brand colors
    "primary":     (16, 185, 129),   # Green - #10B981
    "secondary":   (14, 165, 233),   # Blue - #0EA5E9
    "accent":      (255, 107, 0),    # Orange - #FF6B00
    
    # UI interface colors
    "background":  (13, 17, 23),     # Dark background - #0D1117
    "text":        (230, 237, 243),  # Light text - #E6EDF3
    "panel":       (22, 27, 34),     # Slightly lighter background - #161B22
    "highlight":   (56, 139, 253),   # Highlight blue - #388BFD
    "warning":     (210, 153, 34),   # Warning yellow - #D29922
    "error":       (248, 81, 73),    # Error red - #F85149
    "success":     (63, 185, 80),    # Success green - #3FB950
    
    # Menu-specific enhanced color scheme
    "bg":          (24, 24, 27),     # Dark gray background - #18181B
    
    # Text grays with proper contrast
    "gray100":     (243, 244, 246),  # Lightest text - #F3F4F6
    "gray200":     (229, 231, 235),  # Very light text - #E5E7EB
    "gray300":     (212, 212, 216),  # Light text - #D4D4D8
    "gray400":     (156, 163, 175),  # Medium light - #9CA3AF
    "gray500":     (113, 113, 122),  # Medium - #71717A
    "gray600":     (82, 82, 91),     # Medium dark - #52525B
    "gray700":     (63, 63, 70),     # Dark - #3F3F46
    "gray800":     (39, 39, 42),     # Very dark - #27272A
    
    # Menu options with color harmony
    "option1":     (16, 185, 129),   # Green (same as primary)
    "option2":     (14, 165, 233),   # Blue (same as secondary)
    "option3":     (236, 72, 153),   # Pink - #EC4899
    "option4":     (139, 92, 246),   # Purple - #8B5CF6
    "option5":     (217, 119, 6),    # Amber - #D97706
    "option6":     (6, 182, 212),    # Cyan - #06B6D4
    "option7":     (245, 158, 11),   # Yellow - #F59E0B
    "option8":     (249, 115, 22),   # Orange - #F97316
    
    # Selection indicators
    "selection":    (255, 255, 255), # Pure white for selection
    "hover":        (181, 181, 204), # Light lavender for hover
    "inactive":     (120, 120, 135), # Muted color for inactive
}

# Predefined color palettes
_PALETTES: Dict[str, Dict[str, ColorType]] = {
    "default": _ROLE_COLORS,
    
    "monochrome": {
        "primary":    (200, 200, 200),
        "secondary":  (150, 150, 150),
        "accent":     (100, 100, 100),
        "background": (25, 25, 25),
        "text":       (225, 225, 225),
        "panel":      (50, 50, 50),
        "highlight":  (175, 175, 175),
        "warning":    (175, 150, 100),
        "error":      (200, 100, 100),
        "success":    (100, 200, 100),
        
        # Menu mappings for monochrome
        "gray100":    (230, 230, 230),
        "gray200":    (210, 210, 210),
        "gray300":    (190, 190, 190),
        "gray400":    (170, 170, 170),
        "gray500":    (150, 150, 150),
        "gray600":    (120, 120, 120),
        "gray700":    (90, 90, 90),
        "gray800":    (60, 60, 60),
        
        "option1":    (220, 220, 220),
        "option2":    (200, 200, 200),
        "option3":    (180, 180, 180),
        "option4":    (160, 160, 160),
        "option5":    (140, 140, 140),
        "option6":    (120, 120, 120),
        "option7":    (100, 100, 100),
        "option8":    (80, 80, 80),
        
        "selection":  (255, 255, 255),
        "hover":      (200, 200, 200),
        "inactive":   (150, 150, 150),
    },
    
    "high_contrast": {
        "primary":    (0, 255, 0),
        "secondary":  (0, 0, 255),
        "accent":     (255, 255, 0),
        "background": (0, 0, 0),
        "text":       (255, 255, 255),
        "panel":      (40, 40, 40),
        "highlight":  (0, 255, 255),
        "warning":    (255, 255, 0),
        "error":      (255, 0, 0),
        "success":    (0, 255, 0),
        
        # Menu mappings for high contrast
        "gray100":    (255, 255, 255),
        "gray200":    (240, 240, 240),
        "gray300":    (220, 220, 220),
        "gray400":    (200, 200, 200),
        "gray500":    (180, 180, 180),
        "gray600":    (150, 150, 150),
        "gray700":    (120, 120, 120),
        "gray800":    (90, 90, 90),
        
        "option1":    (0, 255, 0),
        "option2":    (255, 255, 0),
        "option3":    (255, 0, 255),
        "option4":    (0, 255, 255),
        "option5":    (255, 128, 0),
        "option6":    (128, 255, 0),
        "option7":    (0, 128, 255),
        "option8":    (255, 0, 128),
        
        "selection":  (255, 255, 255),
        "hover":      (200, 200, 200),
        "inactive":   (150, 150, 150),
    }
}

# Current active palette
_current_palette = "default"

def get_color(role: str) -> ColorType:
    """
    Return the RGB tuple for a color role from the current palette.
    
    Args:
        role: A color role name (e.g., "primary", "secondary", "accent", etc.)
        
    Returns:
        A tuple of (r, g, b) values
        
    Raises:
        ValueError: If the color role is unknown
    """
    try:
        return _PALETTES[_current_palette][role]
    except KeyError:
        if role in _ROLE_COLORS:
            return _ROLE_COLORS[role]
        raise ValueError(f"Unknown color role: {role}")

def set_palette(palette_name: str) -> None:
    """
    Change the active color palette.
    
    Args:
        palette_name: Name of the palette to use ("default", "monochrome", "high_contrast")
        
    Raises:
        ValueError: If the palette name is unknown
    """
    global _current_palette
    if palette_name in _PALETTES:
        _current_palette = palette_name
    else:
        raise ValueError(f"Unknown palette: {palette_name}")

def get_current_palette() -> str:
    """Return the name of the currently active palette."""
    return _current_palette

def rgb_to_hex(color: ColorType) -> str:
    """Convert an RGB tuple to a hex color string."""
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

def hex_to_rgb(hex_color: str) -> ColorType:
    """
    Convert a hex color string to an RGB tuple.
    
    Args:
        hex_color: Hex color in format "#RRGGBB" or "RRGGBB"
        
    Returns:
        RGB tuple (r, g, b)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def lighten(color: Union[str, ColorType], amount: float = 0.1) -> ColorType:
    """
    Lighten a color by the specified amount.
    
    Args:
        color: Color role name or RGB tuple
        amount: Value between 0 and 1 representing how much to lighten
        
    Returns:
        Lightened RGB tuple
    """
    if isinstance(color, str):
        rgb = get_color(color)
    else:
        rgb = color
    
    return tuple(min(255, int(c + (255 - c) * amount)) for c in rgb)

def darken(color: Union[str, ColorType], amount: float = 0.1) -> ColorType:
    """
    Darken a color by the specified amount.
    
    Args:
        color: Color role name or RGB tuple
        amount: Value between 0 and 1 representing how much to darken
        
    Returns:
        Darkened RGB tuple
    """
    if isinstance(color, str):
        rgb = get_color(color)
    else:
        rgb = color
    
    return tuple(max(0, int(c * (1 - amount))) for c in rgb)

def get_complementary(color: Union[str, ColorType]) -> ColorType:
    """
    Get the complementary color.
    
    Args:
        color: Color role name or RGB tuple
        
    Returns:
        Complementary RGB tuple
    """
    if isinstance(color, str):
        rgb = get_color(color)
    else:
        rgb = color
    
    return tuple(255 - c for c in rgb)

def register_color(role: str, color: ColorType, palette: Optional[str] = None) -> None:
    """
    Register a new color or override an existing one.
    
    Args:
        role: Color role name
        color: RGB tuple
        palette: Optional palette name. If None, adds to all palettes
    """
    if palette is None:
        _ROLE_COLORS[role] = color
        for pal in _PALETTES.values():
            pal[role] = color
    elif palette in _PALETTES:
        _PALETTES[palette][role] = color
    else:
        raise ValueError(f"Unknown palette: {palette}")

def create_palette(name: str, colors: Dict[str, ColorType]) -> None:
    """
    Create a new color palette.
    
    Args:
        name: Palette name
        colors: Dictionary of color roles to RGB tuples
    """
    if name in _PALETTES:
        raise ValueError(f"Palette already exists: {name}")
    
    # Start with all colors from the default palette
    new_palette = dict(_ROLE_COLORS)
    # Override with provided colors
    new_palette.update(colors)
    _PALETTES[name] = new_palette
