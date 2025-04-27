# ai_platform_trainer/engine/renderer.py

import pygame
from typing import List
from ai_platform_trainer.engine.entities.entity import Entity

class Renderer:
    def render(self, surface: pygame.Surface, entities: List[Entity]) -> None:
        for entity in entities:
            entity.draw(surface)
