import random
from ai_platform_trainer.core.color_manager import get_color, darken, lighten

from ai_platform_trainer.engine.entities.enemy import Enemy
from ai_platform_trainer.engine.entities.player import Player


class EntityFactory:
    @staticmethod
    def create_player() -> Player:
        """Create a player entity with primary color."""
        return Player(position=(400, 300), color=get_color("primary"), speed=5)
    
    @staticmethod
    def create_enemies(count: int = 5) -> list[Enemy]:
        """
        Create a group of enemy entities with secondary color variations.
        
        Args:
            count: Number of enemies to create
            
        Returns:
            A list of Enemy instances
        """
        enemies = []
        
        for i in range(count):
            # Create varied enemy colors by lightening/darkening the secondary color
            # This gives visual variety while maintaining the color theme
            variation = (i / count) * 0.4 - 0.2  # Range from -0.2 to +0.2
            
            if variation < 0:
                color = darken("secondary", abs(variation))
            else:
                color = lighten("secondary", variation)
                
            enemies.append(
                Enemy(
                    position=(random.randint(50, 750), random.randint(50, 550)),
                    color=color,
                    radius=10
                )
            )
            
        return enemies
    
    @staticmethod
    def set_entity_theme_colors(entities: list, use_accent: bool = False) -> None:
        """
        Update the colors of existing entities to match the current theme.
        
        Args:
            entities: List of entities to update
            use_accent: If True, use accent color for some entities for contrast
        """
        for i, entity in enumerate(entities):
            if isinstance(entity, Player):
                entity.color = get_color("primary")
            elif isinstance(entity, Enemy):
                if use_accent and i % 3 == 0:  # Every third enemy gets accent color
                    entity.color = get_color("accent")
                else:
                    # Create slight variations in secondary color
                    variation = (i / len(entities)) * 0.3 - 0.15
                    if variation < 0:
                        entity.color = darken("secondary", abs(variation))
                    else:
                        entity.color = lighten("secondary", variation)
