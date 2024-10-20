import json
import os


class DataManager:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_collision_data(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                try:
                    return json.load(file)
                except json.JSONDecodeError:
                    return []
        return []

    def save_collision_data(self, data):
        with open(self.file_path, "w") as file:
            json.dump(data, file, indent=4)
