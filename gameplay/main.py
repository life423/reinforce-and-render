from gameplay.game_manager import Game

if __name__ == "__main__":
    # from core.config import Config

    mode = ["training", "play"]

    game = Game()
    game.run(mode[0])
