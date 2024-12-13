import json
import os


class DataLogger:
    def __init__(self, filename):
        """
        Initialize the DataLogger and create a new empty JSON file.
        If the file exists, it will be overwritten.
        """
        self.filename = filename
        self.data = []  # Initialize the data list

        # Start fresh by creating or overwriting the file
        with open(self.filename, "w") as f:
            json.dump(self.data, f, indent=4)

    def log(self, data_point):
        """
        Add a data point to the internal list of logged data.
        """
        self.data.append(data_point)

    def save(self):
        """
        Save the logged data to the specified file.
        """
        with open(self.filename, "w") as f:
            json.dump(self.data, f, indent=4)
