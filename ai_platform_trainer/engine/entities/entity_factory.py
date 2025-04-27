import random
from ai_platform_trainer.core.color_manager import get_color

from ai_platform_trainer.engine.entities.enemy import Enemy
from ai_platform_trainer.engine.entities.player import Player


class EntityFactory:
    @staticmethod
    def create_player() -> Player:
        return Player(position=(400, 300), color=get_color("primary"), speed=5)

    @staticmethod
    def create_enemies(count: int = 5) -> list[Enemy]:
        return [
            Enemy(
                position=(random.randint(50, 750), random.randint(50, 550)),
                color=get_color("secondary"),
                radius=10
            )
            for _ in range(count)
        ]
