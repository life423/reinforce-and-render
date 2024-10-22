import os
import time
import json
import math
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get MongoDB URI from environment variables
mongo_uri = os.getenv('MONGO_URI')

# Establish a connection to the MongoDB server
client = MongoClient(mongo_uri)

# Access the specific database and collection
db = client['pixel_pursuit_db']
training_data_collection = db['training_data']


def calculate_distance(player_position, enemy_position):
    return math.sqrt((player_position['x'] - enemy_position['x']) ** 2 +
                     (player_position['y'] - enemy_position['y']) ** 2)


def insert_training_data(player_position, enemy_position, collision):
    # Calculate the distance from the enemy to the player
    distance = calculate_distance(player_position, enemy_position)

    # Create a document to insert
    document = {
        "timestamp": time.time(),
        "player_position": player_position,
        "enemy_position": enemy_position,
        "collision": collision,
        "distance": distance
    }

    # Insert the document into the collection
    result = training_data_collection.insert_one(document)

    # Print the ID of the inserted document
    print(f"Inserted document ID: {result.inserted_id}")


# Example usage
player_position = {"x": 100, "y": 150}
enemy_position = {"x": 200, "y": 250}
collision = False

insert_training_data(player_position, enemy_position, collision)
