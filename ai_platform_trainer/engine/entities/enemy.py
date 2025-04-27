# enemy.py
import random
from ai_platform_trainer.engine.entities.entity import Entity

class Enemy(Entity):
    def update(self, actions: dict) -> None:
        # simple wandering AI
        self.x += random.choice([-1, 0, 1]) * 2
        self.y += random.choice([-1, 0, 1]) * 2
