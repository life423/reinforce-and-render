import json
import os

class DataLogger:
    def __init__(self, filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.basename(filename)  # strip directories from filename
        self.filename = os.path.join(base_dir, '../../data/raw', filename)
        self.data = []
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

    def log(self, data_point):
        self.data.append(data_point)

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f)