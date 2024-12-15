# ai_platform_trainer/gameplay/modes/play_mode.py
import logging


class PlayModeManager:
    def __init__(self, game):
        self.game = game

    def update(self, current_time: int) -> None:
        if not self.game.player.handle_input():
            logging.info("Player requested to quit. Exiting game.")
            self.game.running = False
            return

        try:
            self.game.enemy.update_movement(
                self.game.player.position["x"],
                self.game.player.position["y"],
                self.game.player.step,
                current_time,
            )
            logging.debug("Enemy movement updated in play mode.")
        except Exception as e:
            logging.error(f"Error updating enemy movement: {e}")
            self.game.running = False
            return

        if self.game.check_collision():
            logging.info("Collision detected!")
            self.game.enemy.hide()
            self.game.is_respawning = True
            self.game.respawn_timer = current_time + self.game.respawn_delay
            logging.info(f"Enemy will respawn in {self.game.respawn_delay} ms.")

        self.game.handle_respawn(current_time)
        if self.game.enemy.fading_in:
            self.game.enemy.update_fade_in(current_time)

        enemy_pos = (
            (self.game.enemy.pos["x"], self.game.enemy.pos["y"])
            if self.game.enemy.visible
            else (0, 0)
        )
        self.game.player.update_missiles(enemy_pos)
        self.game.check_missile_collisions()
