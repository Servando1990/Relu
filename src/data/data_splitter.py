import pandas as pd
from datetime import datetime, timedelta

import numpy as np

class DataSplitter:
    def __init__(self,
                  data: pd.DataFrame,
                  target_variable: str,
                  sku_column: str,
                  date_column: str ,
                  train_months: int,
                  val_months:int =0,
                  test_weeks: int =0):
                # Store the data and settings
        
        self.data = data.copy()
        self.date_column = date_column

        if not isinstance(data, pd.DataFrame):
            raise ValueError("data should be a pandas DataFrame")
        if not self.date_column in data.columns:
            raise ValueError("DataFrame must contain a 'Date' column")
        
        # Convert 'Date' column to datetime
        #data[self.date_column] = pd.to_datetime(data[self.date_column])
        
        self.target_variable = target_variable
        self.sku_column = sku_column
        self.train_months = train_months
        self.val_months = val_months
        self.test_weeks = test_weeks

        # Check for validity of train, validation, test split
        if train_months <= 0 or val_months < 0 or test_weeks < 0:
            raise ValueError("Months and weeks must be non-negative and training months must be positive.")

    def split_data(self):
        #print(f"Debug: First few entries in date column: {self.data[self.date_column].head()}")
        print(f"Debug: Data type of date column before processing: {self.data[self.date_column].dtype}")
        print(f"Debug: Unique dates before processing: {self.data[self.date_column].unique()}")
        # Define date thresholds for train, validation, test split
        max_date = self.data[self.date_column].max()
        train_val_threshold = max_date - pd.DateOffset(months=self.train_months + self.val_months)
        val_test_threshold = max_date - pd.DateOffset(weeks=self.test_weeks)
        
        # Split the data into training, validation and testing sets
        train_data = self.data[self.data[self.date_column] <= train_val_threshold]
        val_data = self.data[(self.data[self.date_column] > train_val_threshold) & (self.data[self.date_column] <= val_test_threshold)]
        test_data = self.data[self.data[self.date_column] > val_test_threshold]

        train_skus = set(train_data[self.sku_column])
        val_skus = set(val_data[self.sku_column])
        test_skus = set(test_data[self.sku_column])

        print(f"Number of SKUs in train: {len(train_skus)}")
        print(f"Number of SKUs in validation: {len(val_skus)}")
        print(f"Number of SKUs in test: {len(test_skus)}")

        print(f"Min date in dataset: {self.data[self.date_column].min()}")
        print(f"Max date in dataset: {self.data[self.date_column].max()}")
        print(f"Train-Val threshold: {train_val_threshold}")
        print(f"Val-Test threshold: {val_test_threshold}")


        # Get the set of SKUs that are present in both the train_data, val_data and test_data
        train_val_test_skus = set(train_data[self.sku_column]).intersection(set(val_data[self.sku_column])).intersection(set(test_data[self.sku_column]))

        # Filter the train, val and test data to only include the SKUs that are present in all three sets
        train_data = train_data[train_data[self.sku_column].isin(train_val_test_skus)]
        val_data = val_data[val_data[self.sku_column].isin(train_val_test_skus)]
        test_data = test_data[test_data[self.sku_column].isin(train_val_test_skus)]

        # Add a warning message if there are no common SKUs or how many SKUs were removed
        if len(train_val_test_skus) == 0:
            raise ValueError("No common SKUs in the train, validation and test sets. Try increasing the train months or decreasing the validation months or test weeks.")
        elif len(train_val_test_skus) < len(self.data[self.sku_column].unique()):
            print(f"Warning: {len(self.data[self.sku_column].unique()) - len(train_val_test_skus)} SKUs were removed from the train, validation and test sets because they were not present in all three sets.")

        # Return the training, validation, testing datasets
        X_train, y_train = train_data.drop(columns=[self.target_variable, self.date_column]), train_data[self.target_variable]
        X_val, y_val = val_data.drop(columns=[self.target_variable, self.date_column]), val_data[self.target_variable]
        X_test, y_test = test_data.drop(columns=[self.target_variable, self.date_column]), test_data[self.target_variable]

        print(f"Training data covers from {min(train_data[self.date_column])} to {max(train_data[self.date_column])}")
        print(f"Validation data covers from {min(val_data[self.date_column])} to {max(val_data[self.date_column])}")
        print(f"Test data covers from {min(test_data[self.date_column])} to {max(test_data[self.date_column])}")

        return X_train, y_train, X_val, y_val, X_test, y_test


    


