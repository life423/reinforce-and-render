import json
import os


class DataLogger:
    def __init__(self, filename: str):
        """
        Initialize the DataLogger.
        If the file exists, delete it.
        Then create a new empty JSON file.

        :param filename: Path to the JSON file where data will be logged
        """
        self.filename = filename

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        # If file exists, remove it
        if os.path.exists(self.filename):
            try:
                os.remove(self.filename)
            except OSError as e:
                print(f"Error deleting existing file {self.filename}: {e}")

        # Create a new empty JSON file
        self.data = []
        try:
            with open(self.filename, "w") as f:
                json.dump(self.data, f, indent=4)
        except IOError as e:
            print(f"Error creating file {self.filename}: {e}")

    def log(self, data_point: dict) -> None:
        """
        Add a data point to the internal list of logged data.

        :param data_point: Dictionary containing data to log
        """
        self.data.append(data_point)

    def save(self) -> None:
        """
        Save the logged data to the specified file.
        """
        try:
            with open(self.filename, "w") as f:
                json.dump(self.data, f, indent=4)
        except IOError as e:
            print(f"Error saving data to {self.filename}: {e}")
