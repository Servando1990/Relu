import pandas as pd
from datetime import datetime, timedelta

class DataSplitter:
    def __init__(self, data, train_months, test_months=None, test_weeks=None):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data should be a pandas DataFrame")
        if not 'Date' in data.columns:
            raise ValueError("DataFrame must contain a 'Date' column")
        if not 'SKU' in data.columns:
            raise ValueError("DataFrame must contain a 'SKU' column")
        
        # Convert 'Date' column to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Store the data and settings
        self.data = data
        self.train_months = train_months
        self.test_months = test_months
        self.test_weeks = test_weeks
        
        # Log information about the dataset
        min_date = self.data['Date'].min()
        max_date = self.data['Date'].max()
        total_months = (max_date.year - min_date.year) * 12 + max_date.month - min_date.month
        print(f"Min Date: {min_date}, Max Date: {max_date}, Total Months: {total_months}")

    def split_data(self):
        # Calculate the split date
        max_date = self.data['Date'].max()
        split_date = max_date - pd.DateOffset(months=self.train_months)

        # Split the data into training and testing sets
        train_data = self.data[self.data['Date'] <= split_date].copy()
        test_data = self.data[self.data['Date'] > split_date].copy()

        # Get the set of SKUs that are present in both the training and test sets
        common_skus = set(train_data['SKU']).intersection(set(test_data['SKU']))
        # Filter the training and test sets to only include the common SKUs
        train_data = train_data[train_data['SKU'].isin(common_skus)]
        test_data = test_data[test_data['SKU'].isin(common_skus)]
        #TODO add a warning message if there are no common SKUs or how many SKUs were removed

        
        # Log data types info
        print("Train data types:")
        print(train_data.dtypes)
        print("\nTest data types:")
        print(test_data.dtypes)

        # Return the training and testing datasets
        return train_data, test_data
    


