import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# Get the MongoDB connection string from environment variable
mongo_uri = os.getenv('MONGO_URI')

# Create a MongoClient object
client = MongoClient(mongo_uri)

# Access the specific database and collection
db = client['pixel_pursuit']  # Use 'pixel_pursuit' as the database name
collection = db['training_data']  # Use 'training_data' as the collection name

# Now you can use 'collection' to insert, find, update, or delete documents
