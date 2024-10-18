import sys
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


def add_project_root_to_sys_path() -> None:
    """
    Add the project root directory to sys.path to allow imports from the entire project.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        

