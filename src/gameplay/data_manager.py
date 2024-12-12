# Removed the DataLogger class entirely since it's redundant
import time
import json
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

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


class TrainingDataHandler:
    def __init__(self, data_file_path):
        # We'll assume this file is still where offline data is stored
        # If we no longer need local logging here, we can remove these references.
        self.data_file_path = data_file_path
        self.mongo_handler = MongoDBHandler()

    def load_data(self):
        # Load collision data from JSON file
        with open(self.data_file_path, 'r') as file:
            collision_data = json.load(file)
        return collision_data

    def save_data_to_mongo(self):
        # Load data from file and save to MongoDB
        collision_data = self.load_data()
        for data_point in collision_data:
            self.mongo_handler.insert_data(data_point)

    def fetch_data_for_training(self):
        # Retrieve all data from MongoDB for training
        return self.mongo_handler.get_training_data()