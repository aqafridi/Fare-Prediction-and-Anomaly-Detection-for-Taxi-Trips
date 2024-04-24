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
        data_types = {
            # 'VendorID': 'int32',
            # 'passenger_count': 'int8',
            'trip_distance': 'float32',
            # 'RatecodeID': 'int8',
            'store_and_fwd_flag': 'object',
            # 'PULocationID': 'int32',
            # 'DOLocationID': 'int32',
            # 'payment_type': 'int8',
            'fare_amount': 'float32',
            'extra': 'float32',
            'mta_tax': 'float32',
            'tip_amount': 'float32',
            'tolls_amount': 'float32',
            'improvement_surcharge': 'float32',
            'total_amount': 'float32',
            'congestion_surcharge': 'float32'
        }

        date_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']

        # Define a converter function to convert string integers to integers
        def str_to_int(x):
            try:
                return int(x)
            except ValueError:
                return None

        # Add converters for columns with string integers
        converters = {'VendorID': str_to_int, 'passenger_count': str_to_int, 'RatecodeID': str_to_int, 'payment_type': str_to_int, 'PULocationID':str_to_int, 'DOLocationID':str_to_int}
        
        return pd.read_csv(self.data_path, dtype=data_types, parse_dates=date_cols, converters=converters)
