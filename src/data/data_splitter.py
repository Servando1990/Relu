import pandas as pd
from datetime import datetime, timedelta

import numpy as np

class DataSplitter:
    def __init__(self, data, target_variable, train_months, val_months=0, test_weeks=0):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data should be a pandas DataFrame")
        if not 'Date' in data.columns:
            raise ValueError("DataFrame must contain a 'Date' column")
        
        # Convert 'Date' column to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Store the data and settings
        self.data = data
        self.target_variable = target_variable
        self.train_months = train_months
        self.val_months = val_months
        self.test_weeks = test_weeks

        # Check for validity of train, validation, test split
        if train_months <= 0 or val_months < 0 or test_weeks < 0:
            raise ValueError("Months and weeks must be non-negative and training months must be positive.")

    def split_data(self):
        # Define date thresholds for train, validation, test split
        max_date = self.data['Date'].max()
        train_val_threshold = max_date - pd.DateOffset(months=self.train_months + self.val_months)
        val_test_threshold = max_date - pd.DateOffset(weeks=self.test_weeks)
        
        # Split the data into training, validation and testing sets
        train_data = self.data[self.data['Date'] <= train_val_threshold]
        val_data = self.data[(self.data['Date'] > train_val_threshold) & (self.data['Date'] <= val_test_threshold)]
        test_data = self.data[self.data['Date'] > val_test_threshold]

        # Return the training, validation, testing datasets
        X_train, y_train = train_data.drop(columns=self.target_variable), train_data[self.target_variable]
        X_val, y_val = val_data.drop(columns=self.target_variable), val_data[self.target_variable]
        X_test, y_test = test_data.drop(columns=self.target_variable), test_data[self.target_variable]

        print(f"Training data covers from {min(train_data['Date'])} to {max(train_data['Date'])}")
        print(f"Validation data covers from {min(val_data['Date'])} to {max(val_data['Date'])}")
        print(f"Test data covers from {min(test_data['Date'])} to {max(test_data['Date'])}")

        return X_train, y_train, X_val, y_val, X_test, y_test


    


