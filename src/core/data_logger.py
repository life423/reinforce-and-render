import os
from pymongo import MongoClient
import time
import json

class DataLogger:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.collision_data = []
        mongo_uri = os.getenv('MONGO_URI')
        self.client = MongoClient(mongo_uri)
        self.db = self.client["pixel_pursuit_db"]
        self.collection = self.db["training_data"]

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

        # Append to collision data list for local saving
        self.collision_data.append(data_point)

        # Save to MongoDB
        self.collection.insert_one(data_point)
        print("Data inserted into MongoDB")

    def save_data(self):
        # Check if the file exists
        if os.path.exists(self.data_file_path):
            # Load the existing data
            with open(self.data_file_path, 'r') as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = []  # If the file is empty or corrupted
        else:
            # If the file doesn't exist, initialize as an empty list
            existing_data = []

        # Append new collision data
        existing_data.extend(self.collision_data)

        # Save the updated data back to the JSON file
        with open(self.data_file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)

        # Clear the in-memory collision data after saving
        self.collision_data = []
