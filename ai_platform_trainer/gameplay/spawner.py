# ai_platform_trainer/gameplay/spawner.py
import logging
from ai_platform_trainer.gameplay.utils import find_valid_spawn_position
from ai_platform_trainer.gameplay.config import config


def spawn_entities(game):
    if not game.player or not game.enemy:
        logging.error("Entities not initialized properly.")
        game.running = False
        return

    player_pos = find_valid_spawn_position(
        screen_width=game.screen_width,
        screen_height=game.screen_height,
        entity_size=game.player.size,
        margin=config.WALL_MARGIN,
        min_dist=0,
        other_pos=None,
    )

    enemy_pos = find_valid_spawn_position(
        screen_width=game.screen_width,
        screen_height=game.screen_height,
        entity_size=game.enemy.size,
        margin=config.WALL_MARGIN,
        min_dist=config.MIN_DISTANCE,
        other_pos=(game.player.position["x"], game.player.position["y"]),
    )

    game.player.position["x"], game.player.position["y"] = player_pos
    game.enemy.pos["x"], game.enemy.pos["y"] = enemy_pos

    logging.info(f"Spawned player at {player_pos} and enemy at {enemy_pos}.")


def respawn_enemy_at_position(game):
    new_pos = find_valid_spawn_position(
        screen_width=game.screen_width,
        screen_height=game.screen_height,
        entity_size=game.enemy.size,
        margin=config.WALL_MARGIN,
        min_dist=config.MIN_DISTANCE,
        other_pos=(game.player.position["x"], game.player.position["y"]),
    )
    game.enemy.set_position(new_pos[0], new_pos[1])
    return new_pos
