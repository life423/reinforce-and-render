# ai_platform_trainer/gameplay/utils/entity_utils.py

def wrap_position(x: float, y: float, width: float, height: float, entity_size: float) -> Tuple[float, float]:
    def wrap(val: float, lower: float, upper: float) -> float:
        if val < lower:
            return upper
        elif val > upper:
            return lower
        return val

    new_x = wrap(x, -entity_size, width)
    new_y = wrap(y, -entity_size, height)
    return new_x, new_y