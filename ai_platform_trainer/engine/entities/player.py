# player.py
from ai_platform_trainer.engine.entities.entity import Entity

class Player(Entity):
    def update(self, actions: dict) -> None:
        if actions.get("up"):
            self.y -= 5
        if actions.get("down"):
            self.y += 5
        if actions.get("left"):
            self.x -= 5
        if actions.get("right"):
            self.x += 5
