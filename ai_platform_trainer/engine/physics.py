import pymunk
from typing import Tuple

class PhysicsSystem:
    def __init__(self, arena_width: int = 800, arena_height: int = 600):
        """
        Initialize the physics system with a space and walls.
        
        Args:
            arena_width: Width of the arena in pixels
            arena_height: Height of the arena in pixels
        """
        # Create a new space with gravity (0, 0) - no gravity for top-down game
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        
        # Arena boundaries (static walls)
        self._add_walls(arena_width, arena_height)
        
    def _add_walls(self, width: int, height: int) -> None:
        """
        Create static walls around the arena.
        
        Args:
            width: Arena width
            height: Arena height
        """
        # Create static body for walls
        static_body = self.space.static_body
        
        # Create wall segments (top, right, bottom, left)
        walls = [
            # Top wall: (left, top) to (right, top)
            [(0, 0), (width, 0)],
            # Right wall: (right, top) to (right, bottom)
            [(width, 0), (width, height)],
            # Bottom wall: (right, bottom) to (left, bottom)
            [(width, height), (0, height)],
            # Left wall: (left, bottom) to (left, top)
            [(0, height), (0, 0)]
        ]
        
        # Add each wall segment to space
        for wall in walls:
            segment = pymunk.Segment(static_body, wall[0], wall[1], 0)
            segment.elasticity = 1.0  # Perfect bounce
            segment.friction = 0.5
            self.space.add(segment)
    
    def create_player_body(self, position: Tuple[int, int], radius: int, mass: float = 10.0) -> pymunk.Body:
        """
        Create a player physics body.
        
        Args:
            position: Initial position (x, y)
            radius: Radius of the player circle in pixels
            mass: Mass of the player body
            
        Returns:
            The created pymunk.Body
        """
        # Create dynamic body
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = position
        
        # Create circle shape for collision
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 1.0  # Perfect bounce
        shape.friction = 0.5
        
        # Add to space
        self.space.add(body, shape)
        
        return body
    
    def create_enemy_body(self, position: Tuple[int, int], radius: int, mass: float = 5.0) -> pymunk.Body:
        """
        Create an enemy physics body.
        
        Args:
            position: Initial position (x, y)
            radius: Radius of the enemy circle in pixels
            mass: Mass of the enemy body
            
        Returns:
            The created pymunk.Body
        """
        # Create dynamic body
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = position
        
        # Create circle shape for collision
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 1.0  # Perfect bounce
        shape.friction = 0.5
        
        # Add to space
        self.space.add(body, shape)
        
        return body
    
    def step_space(self, dt: float) -> None:
        """
        Step the physics simulation forward.
        
        Args:
            dt: Time step in seconds
        """
        # Typically use a fixed time step for stability
        self.space.step(dt)
