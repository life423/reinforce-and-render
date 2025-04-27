# ai_platform_trainer/core/color_manager.py

from typing import Tuple

# Triadic roles mapping to your brand’s 500-level Tailwind shades
_ROLE_COLORS: dict[str, Tuple[int, int, int]] = {
    "primary":   (16, 185, 129),  # brandGreen[500]  #10B981
    "secondary": (14, 165, 233),  # brandBlue[500]   #0EA5E9
    "accent":    (255, 107,   0),  # brandOrange[500] #FF6B00
}

def get_color(role: str) -> Tuple[int, int, int]:
    """
    Return the RGB tuple for one of: "primary", "secondary", "accent".
    """
    try:
        return _ROLE_COLORS[role]
    except KeyError:
        raise ValueError(f"Unknown color role: {role}")
