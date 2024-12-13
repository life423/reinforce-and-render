import json
import os


class DataLogger:
    def __init__(self, filename):
        """
        Initialize the DataLogger.
        If the file exists, delete it.
        Then create a new empty JSON file.
        """
        self.filename = filename

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        # If file exists, remove it
        if os.path.exists(self.filename):
            os.remove(self.filename)

        # Create a new empty JSON file
        self.data = []
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
