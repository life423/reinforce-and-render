from db_connection import client  # Assuming db_connection sets up the client

# Define the database and collection names
database_name = "pixel_pursuit_db"
collection_name = "training_data"

# Create the database and collection if they do not exist
db = client[database_name]

# Check if the collection exists, create if it doesn't
if collection_name not in db.list_collection_names():
    collection = db.create_collection(collection_name)
    print(f"Collection '{collection_name}' created.")
else:
    collection = db[collection_name]
    print(f"Using existing collection '{collection_name}'.")
