import random
from typing import Tuple
import numpy as np
import pymunk

from ai_platform_trainer.engine.physics import create_physics_space, step_space

# Constants
ARENA_W, ARENA_H = 800, 600
PLAYER_RADIUS     = 15
ENEMY_RADIUS      = 10
ENEMY_SPEED       = 350.0        # pixels/sec
TIME_STEP         = 1.0 / 60.0
SUBSTEPS          = 4            # physics sub-steps per frame
MAX_STEPS         = 600

class RLBounceEnv:
    """
    A single-agent environment:
    - Agent = ONE enemy trying to hit the player
    - Other enemies are static or random (we'll start without them)
    """

    def __init__(self, seed: int | None = None):
        self.rng   = random.Random(seed)
        self.space = create_physics_space(gravity=(0, 0))
        # square walls already added
        self.player_body: pymunk.Body | None = None
        self.enemy_body:  pymunk.Body | None = None
        self.steps  = 0

        # pre-allocate observation buffer (6,)
        self._obs_buf = np.zeros(6, dtype=np.float32)

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        """Spawn bodies and return initial observation."""
        self.space.remove(*self.space.bodies)
        self.steps = 0

        # --- player ---
        px, py = self._random_pos()
        self.player_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.player_body.position = (px, py)
        shape_p = pymunk.Circle(self.player_body, PLAYER_RADIUS)
        shape_p.elasticity = 1.0
        self.space.add(self.player_body, shape_p)

        # --- enemy (learning agent) ---
        ex, ey = self._far_random_pos(px, py, min_dist=150)
        self.enemy_body = pymunk.Body(mass=1.0, moment=10.0)
        self.enemy_body.position = (ex, ey)
        shape_e = pymunk.Circle(self.enemy_body, ENEMY_RADIUS)
        shape_e.elasticity = 1.0
        self.space.add(self.enemy_body, shape_e)

        return self._get_obs()

    # ------------------------------------------------------------------ #
    def step(self, action: Tuple[float, float]):
        """
        action: (dx, dy) floats in [-1,1].
        Returns: obs, reward, done, info
        """
        # --- clip & normalize action ---
        dx, dy = np.clip(action, -1.0, 1.0)
        vec = np.array([dx, dy], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        vx, vy = vec * ENEMY_SPEED
        self.enemy_body.velocity = (vx, vy)

        # --- physics sub-steps ---
        for _ in range(SUBSTEPS):
            step_space(self.space, TIME_STEP / SUBSTEPS)

        self.steps += 1

        # --- reward ---
        done = False
        reward = -0.01  # time penalty

        dist = self._distance()
        reward += 1.0 / (1.0 + dist)   # dense reward

        if dist < (PLAYER_RADIUS + ENEMY_RADIUS):
            reward += 10.0
            done = True

        if self.steps >= MAX_STEPS:
            done = True

        return self._get_obs(), reward, done, {}

    # ------------------------------------------------------------------ #
    # -------------------- helpers ------------------------------------- #
    def _get_obs(self) -> np.ndarray:
        px, py = self.player_body.position
        ex, ey = self.enemy_body.position
        vx, vy = self.enemy_body.velocity

        # scale positions to [-1,1]
        self._obs_buf[:] = [
            (px - ARENA_W/2) / (ARENA_W/2),
            (py - ARENA_H/2) / (ARENA_H/2),
            (ex - ARENA_W/2) / (ARENA_W/2),
            (ey - ARENA_H/2) / (ARENA_H/2),
            vx / ENEMY_SPEED,
            vy / ENEMY_SPEED,
        ]
        return self._obs_buf

    def _distance(self) -> float:
        return (self.player_body.position - self.enemy_body.position).length

    # pick random pos not too close to walls
    def _random_pos(self, margin: int = 50):
        x = self.rng.uniform(margin, ARENA_W - margin)
        y = self.rng.uniform(margin, ARENA_H - margin)
        return x, y

    def _far_random_pos(self, ox, oy, min_dist: float, tries: int = 100):
        for _ in range(tries):
            x, y = self._random_pos()
            if ((x-ox)**2 + (y-oy)**2) ** 0.5 >= min_dist:
                return x, y
        return self._random_pos()
