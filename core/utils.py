
import os

def clamp_position(value: int, min_value: int, max_value: int) -> int:
    """
    Clamps a value between a minimum and maximum range.

    Args:
        value (int): The value to be clamped.
        min_value (int): The minimum allowed value.
        max_value (int): The maximum allowed value.

    Returns:
        int: The clamped value within the specified range.
    """
    return max(min_value, min(max_value, value))



