import time
import json


class DataLogger:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.collision_data = []

    def log_data(self, player_position, enemy_position):
        # Calculate the distance between player and enemy
        distance = ((player_position["x"] - enemy_position["x"]) **
                    2 + (player_position["y"] - enemy_position["y"]) ** 2) ** 0.5

        # Create a data point
        data_point = {
            "timestamp": time.time(),
            "player_position": player_position.copy(),
            "enemy_position": enemy_position.copy(),
            "distance": distance
        }

        # Append to collision data list
        self.collision_data.append(data_point)

        # Save to file every 100 frames to prevent performance issues
        if len(self.collision_data) % 100 == 0:
            self.save_data()

    def save_data(self):
        # Save collision data to JSON file
        with open(self.data_file_path, 'w') as file:
            json.dump(self.collision_data, file, indent=4)
