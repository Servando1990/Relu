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
        # Validate that at least test_months or test_weeks is provided
        if self.test_months is None and self.test_weeks is None:
            raise ValueError("Either test_months or test_weeks should be provided")
        
        # Calculate the split date
        max_date = self.data['Date'].max()
        if self.test_months is not None:
            split_date = max_date - pd.DateOffset(months=self.test_months)
        else:
            split_date = max_date - pd.DateOffset(weeks=self.test_weeks)
        
        # Split the data into training and testing sets
        train_data = self.data[self.data['Date'] <= split_date].copy()
        test_data = self.data[self.data['Date'] > split_date].copy()

        # Check that all SKUs in test_data are also present in train_data
        train_skus = set(train_data['SKU'].unique())
        test_skus = set(test_data['SKU'].unique())
        if not test_skus.issubset(train_skus):
            print("Warning: There are SKUs in the test data that are not present in the training data.")
        
        # Log data types info
        print("Train data types:")
        print(train_data.dtypes)
        print("\nTest data types:")
        print(test_data.dtypes)

        # Return the training and testing datasets
        return train_data, test_data

