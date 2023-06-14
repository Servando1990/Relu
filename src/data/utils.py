import numpy as np


class Utils:
    def convert_to_float(value):
        if isinstance(value, float):
            return value
        try:
            return float(value.replace(',', '.'))
        except ValueError:
            return np.nan
        
    # create a function that adds "_" in every space for all columns in a dataframe
    def add_underscore(df):
        df.columns = df.columns.str.replace(' ', '_')
        return df
    