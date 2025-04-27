# ai_platform_trainer/core/color_manager.py

from typing import Tuple, Dict, Union, Optional

ColorType = Tuple[int, int, int]

# Triadic roles mapping to your brand's 500-level Tailwind shades
_ROLE_COLORS: Dict[str, ColorType] = {
    # Core brand colors
    "primary":    (16, 185, 129),  # brandGreen[500]  #10B981
    "secondary":  (14, 165, 233),  # brandBlue[500]   #0EA5E9
    "accent":     (255, 107, 0),   # brandOrange[500] #FF6B00
    
    # UI colors
    "background": (13, 17, 23),    # Dark background
    "text":       (230, 237, 243), # Light text
    "panel":      (22, 27, 34),    # Slightly lighter than background
    "highlight":  (56, 139, 253),  # Highlight blue
    "warning":    (210, 153, 34),  # Warning yellow
    "error":      (248, 81, 73),   # Error red
    "success":    (63, 185, 80),   # Success green
    
    # Gray scale for menu
    "bg":         (24, 24, 27),    # Dark gray background
    "gray300":    (212, 212, 216), # Light gray for text
    "gray500":    (113, 113, 122), # Medium gray for secondary text
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
