import pandas as pd
from math import ceil

class Dataloader:
    def __init__(self, file_path, batch_size=500):
        """
        Initialize the Dataloader.

        Args:
        file_path (str): The path to the CSV file.
        batch_size (int): The size of each batch to be read.
        """
        self.file_path = file_path
        self.batch_size = batch_size
        self.current_index = 0
        col_names = pd.read_csv(file_path, nrows=1).columns
        self.col_names= col_names
        self.total_rows = sum(1 for line in open(self.file_path)) - 1  # Get total number of rows excluding header
        self.data = None
        self.nr_steps = ceil(self.total_rows/batch_size)
        

    def next(self):
        """
        Read the next batch of data from the CSV file.

        Returns:
        pandas.DataFrame: A DataFrame containing the next batch of data.
        """
        
        self._load_next_batch()
        self.current_index += self.batch_size
        self.data.columns= self.col_names
        return self.data

    def hasNext(self):
        """
        Check if there is more data to be read from the CSV file.

        Returns:
        bool: True if there is more data, False otherwise.
        """
        return self.current_index < self.total_rows

    def restart(self):
        """
        Restart reading the CSV file from the beginning.
        """
        self.current_index = 0
        self.data = None

    def _load_next_batch(self):
        """
        Load the next batch of data from the CSV file into memory.
        """
        skiprows = 1 if self.current_index==0 else self.batch_size - 1  # Skip the first n-1 lines
        nr_rows_to_read = min(self.batch_size, self.total_rows - self.current_index)
        self.data = pd.read_csv(self.file_path, skiprows=skiprows, nrows=nr_rows_to_read)
