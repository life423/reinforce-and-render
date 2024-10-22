import time
import json
import os
from pymongo import MongoClient
from dotenv import load_dotenv


from datetime import datetime


class DataLogger:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.collision_data = []

    def log_data(self, player_position, enemy_position):
        # Calculate the distance between player and enemy
        distance = ((player_position["x"] - enemy_position["x"]) **
                    2 + (player_position["y"] - enemy_position["y"]) ** 2) ** 0.5

        # Create a data point with a human-readable timestamp
        data_point = {
            "timestamp": time.time(),  # Unix timestamp
            # Human-readable format
            "formatted_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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


    def load_data(self):
        # Load collision data from JSON file
        with open(self.data_file_path, 'r') as file:
            self.collision_data = json.load(file)

    def get_collision_data(self):
        return self.collision_data

    def set_collision_data(self, collision_data):
        self.collision_data = collision_data


class MongoDBHandler:
    def __init__(self):
        load_dotenv()
        mongo_uri = os.getenv('MONGO_URI')
        self.client = MongoClient(mongo_uri)
        self.db = self.client["pixel_pursuit_db"]
        self.collection = self.db["training_data"]

    def insert_data(self, data):
        self.collection.insert_one(data)
        print("Data inserted successfully into MongoDB.")

    def retrieve_data(self, query):
        return self.collection.find(query)

    def get_training_data(self):
        return list(self.collection.find({}))

    def clear_collection(self):
        self.collection.delete_many({})
        print("Training data collection cleared.")

# Setup the logger and MongoDB handler


class TrainingDataHandler:
    def __init__(self, data_file_path):
        self.logger = DataLogger(data_file_path)
        self.mongo_handler = MongoDBHandler()

    def save_data_to_mongo(self):
        # Load data from file and save to MongoDB
        self.logger.load_data()
        for data_point in self.logger.get_collision_data():
            self.mongo_handler.insert_data(data_point)

    def fetch_data_for_training(self):
        # Retrieve all data from MongoDB for training
        return self.mongo_handler.get_training_data()


# Usage Example
# if __name__ == "__main__":
#     data_file_path = "collision_data.json"
#     training_handler = TrainingDataHandler(data_file_path)

#     # Example: Log data and save it
#     player_position = {"x": 100, "y": 150}
#     enemy_position = {"x": 120, "y": 170}
#     training_handler.logger.log_data(player_position, enemy_position)
#     training_handler.logger.save_data()

#     # Save data to MongoDB
#     training_handler.save_data_to_mongo()

#     # Fetch data for training
#     training_data = training_handler.fetch_data_for_training()
    # print("Training data retrieved from MongoDB:", training_data)
