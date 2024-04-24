

import pandas as pd

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        """
        Load the taxi trip data from the specified path.
        
        Returns:
            pandas.DataFrame: The loaded taxi trip data.
        """
        return pd.read_csv(self.data_path)