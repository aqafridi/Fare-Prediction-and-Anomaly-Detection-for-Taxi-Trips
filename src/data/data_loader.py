import pandas as pd

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        """ Load the taxi trip data from the specified path.
        Returns:
            pandas.DataFrame: The loaded taxi trip data.
        """
        data_types = {
            'trip_distance': 'float32',
            'store_and_fwd_flag': 'object',
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

        def str_to_int(x):
            try:
                return int(x)
            except ValueError:
                return None

        converters = {
            'VendorID': str_to_int,
            'passenger_count': str_to_int,
            'RatecodeID': str_to_int,
            'payment_type': str_to_int,
            'PULocationID': str_to_int,
            'DOLocationID': str_to_int
        }

        data = pd.read_csv(self.data_path, dtype=data_types, converters=converters, low_memory=False)
        data[date_cols] = data[date_cols].apply(pd.to_datetime, format="%Y-%m-%d %H:%M:%S", errors='coerce', infer_datetime_format=True)
        return data