import pandas as pd
from typing import List, Iterable, Tuple

import numpy as np
from sympy import group
import logging





class FeatureEngineeringProcess:
    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # create a file handler
        handler = logging.FileHandler('feature_engineering.log')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)

        # store parameters
        self.params = {}


    def grouped_feature_eng(
        self,
        df: pd.DataFrame,
        group_features: List[str],
        features: List[str],

    ):
        """Transform aggregation based on grouping a set of features

        Args:
            df (pd.DataFrame): DataFrame to be transformed
            group_features (list): List of features selected to group the transformation, ej SKU
            features (list): List of features to be transformed.

        Returns:
            df: Transformed pd.DataFrame
        """
        #TODO: add more numerical transformations
        for feature in features:
            for group_feature in group_features:
                df[f"{group_feature}_{feature}_mean"] = df.groupby(group_feature)[feature].transform("mean")
                df[f"{group_feature}_{feature}_min"] = df.groupby(group_feature)[feature].transform("min")
                df[f"{group_feature}_{feature}_max"] = df.groupby(group_feature)[feature].transform("max")

        return df
    
    def compute_gmv(self, df: pd.DataFrame, window: int = 7):
        """Compute GMV based on the quantity and price of the product

        Args:
            df (pd.DataFrame): DataFrame to be transformed

        Returns:
            df: Transformed pd.DataFrame
        """
        # Sort values by sku and date
        df = df.sort_values(by=["SKU", "Date"])
        df["gmv"] = df["Qty"] * df["price"] #TODO hardcoded variables
        #compute gmv per product for the last N days
        df[f"gmv_last_{window}_days"] = df.groupby("SKU")["gmv"].transform(lambda x: x.rolling(window).sum())


        return df

    def datetime_transform(
        self,
        df: pd.DataFrame,
        date_feature: str,
        features: List[str] = ["month", "day"],
    ):
        """Aggregate datetime features

        Args:
            df (pd.DataFrame):
            date_feature (str): df column to be transformed
            features (List[str]): List of date features to extract from the date_feature column
            country (str): Country code for holiday detection

        Returns:
            df: Dataframe with date transformations, (month, day, year, hour, minute, second, season, holidays)
        """
        seasons = {
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Fall",
        10: "Fall",
        11: "Fall",
        12: "Winter",
        }


        df[date_feature] = pd.to_datetime(df[date_feature])

        if "month" in features:
            df[date_feature + "_month"] = df[date_feature].dt.month
        if "day" in features:
            df[date_feature + "_day"] = df[date_feature].dt.day
        if "day_name" in features:
            df[date_feature + "_day_name"] = df[date_feature].dt.day_name()
        if "day_of_the_year" in features:
            df[date_feature + "_day_of_the_year"] = df[date_feature].dt.dayofyear    
        if "week" in features:
            df[date_feature + "_week"] = df[date_feature].dt.isocalendar().week
        if "year" in features:
            df[date_feature + "_year"] = df[date_feature].dt.year
        if "quarter" in features:
            df[date_feature + "_quarter"] = df[date_feature].dt.quarter
        if "hour" in features:
            df[date_feature + "_hour"] = df[date_feature].dt.hour
        if "minute" in features:
            df[date_feature + "_minute"] = df[date_feature].dt.minute
        if "second" in features:
            df[date_feature + "_second"] = df[date_feature].dt.second
        if "season" in features:
            df[date_feature + "_season"] = df[date_feature].dt.month.map(seasons)
        #if "holidays" in features:
            #print('updated ___')
            #holiday_dates = holidays.CountryHoliday('PL', years=df[date_feature].dt.year)
            #df[date_feature + "_holidays"] = df[date_feature].dt.date.map(holiday_dates).notna().astype(int)

        return df
    

    def price_sales_correlation_features_updated(self, data: pd.DataFrame, 
                                     N: int,
                                     uv_pairs = List[Tuple[float, float]]) -> pd.DataFrame:
        
        """Compute the price-sales correlation features for a given range of u and v values

        Args:
            data (pd.DataFrame): DataFrame to be transformed
            N (int): Lookback period in days
            uv_pairs (List[Tuple[float, float]]): List of tuples with u and v values


        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        
        # Sort data by SKU and date
        data = data.sort_values(by=['SKU', 'Date']) #TODO hardcoded variables

        # Group the data by SKU
        grouped = data.groupby('SKU')

        # Placeholder DataFrame for the results
        results = pd.DataFrame()

        for sku, group in grouped:
            for u, v in uv_pairs:
                # Examples of combinations of pairs:
                #  uv_pairs =[(-5.0, 1.0), (-3.0, 1.0), (-2.0, 1.0), (1.0, 1.0), (-1.0, 0.5), (-1.0, 0.33)]
                    # first pair [-2, 1] (example if this is the tio variant after feature importance)
                        # add more pairs: [-2.2, 1], [-1.8] aroun the top variant
                    # second pair [-3, 1]
                    #[-2.5,1 ]
                    #[-1,1/2]
                    #[1,1]
                # Calculate the numerator part of the formula
                numerator = (group['price']**u * group['sales']**v).rolling(window=N).sum()

                # Calculate the sum of price to the power of u over the last N days
                sum_price_u = (group['price']**u).rolling(window=N).sum()

                # Calculate the sum of sales to the power of v over the last N days
                sum_sales_v = (group['sales']**v).rolling(window=N).sum()

                # Calculate the feature f_corr for this combination of u and v
                f_corr = numerator / (sum_price_u * sum_sales_v)

                # Add the feature to the group
                group[f'f_corr_{u}_{v}'] = f_corr

            # Append the group to the results Dataframe
            results = pd.concat([results, group])
        # log and save parameters of the function
        self.logger.info(f"price_sales_correlation_features_updated: N={N}, uv_pairs={uv_pairs}")
        # save parameters of the function
        self.params['price_sales_correlation_features_updated'] = {'N': N, 'uv_pairs': uv_pairs}

        
        return results
    
    def normalize_features(
            self, 
            data: pd.DataFrame, 
            window_sizes: List[int], 
            long_period: int ) -> pd.DataFrame:
        # Generate docstring
        """ Normalize features for pricing model.  
        Args:
            data (pd.DataFrame): DataFrame
            N (int): represents the number of days for a short-term average. 
            window_sizes (List[int]): List of window sizes to be used for the short-term average.
            long_period (int): represents the number of days over a longer-term average. 
                This is used to calculate the reference price (price0) and reference demand (demand0) which are used to normalize the short-term average.
                For example, if long_period is set to 28, it means that you calculate the average price and average demand over the last 28 days to use as the reference values for normalization.
        Returns:
            data (pd.DataFrame): DataFrame with normalized features.
        """
        
        
        # Sort data by SKU and date
        data = data.sort_values(by=['SKU', 'Date'])

        grouped = data.groupby('SKU')

        # Placeholder list for the results
        results = []

        for sku, group in grouped:               
            
            # Calculate reference price and demand
            price0 = group['price'].rolling(window=long_period).mean()
            demand0 = group['sales'].rolling(window=long_period).mean()

            # Use ffill  the NaNs with the last valid observation
            price0 = price0.fillna(method='ffill')
            demand0 = demand0.fillna(method='ffill')

            # Replace zeros with NaNs after forward filling
            price0 = price0.replace(to_replace=0, value=np.nan)
            demand0 = demand0.replace(to_replace=0, value=np.nan)

            price0.fillna(method='bfill', inplace=True)
            demand0.fillna(method='bfill', inplace=True)

            # Loop through the window sizes
            for N in window_sizes:

                # Calculate average price and sales for the last N days
                group[f'avg_price_last_{N}_days'] = group['price'].rolling(window=N).mean()
                group[f'avg_sales_last_{N}_days'] = group['sales'].rolling(window=N).mean()
    
                # Normalize average price and sales
                group[f'normalized_avg_price_{N}_days'] = group[f'avg_price_last_{N}_days'] / price0
                group[f'normalized_avg_sales_{N}_days'] = group[f'avg_sales_last_{N}_days'] / demand0
                
                # Apply log transformation
                group[f'normalized_log_avg_price_{N}_days'] = np.log(group[f'avg_price_last_{N}_days'] / price0)
                group[f'normalized_log_avg_sales_{N}_days'] = np.log(group[f'avg_sales_last_{N}_days'] / demand0)

                # Normalize standard deviation
                group[f'normalized_std_price_{N}_days'] = group['price'].rolling(window=N).std() / group[f'avg_price_last_{N}_days']
                group[f'normalized_std_sales_{N}_days'] = group['sales'].rolling(window=N).std() / group[f'avg_sales_last_{N}_days']

                #  Drop 'avg_price_last_N_days' and 'avg_sales_last_N_days' columns
                group = group.drop(columns=[f'avg_price_last_{N}_days', f'avg_sales_last_{N}_days'])

            results.append(group)

        # Append the group to the results Dataframe
        results = pd.concat(results)
        # log and save parameters of the function
        self.logger.info(f"normalize_features: window_sizes={window_sizes}, long_period={long_period}")
        # save parameters of the function
        self.params['normalize_features'] = {'window_sizes': window_sizes, 'long_period': long_period}
        
        return results
    
    def filter_stability_periods(self, 
                             data: pd.DataFrame,
                             N: int,
                             threshold: float = 0.05, 
                             removal_percentage: float = 0.98,
                             ) -> pd.DataFrame:
    
        # Sort the data by SKU and date
        data = data.sort_values(by=['SKU', 'Date'])
        
        # Calculate the N-day average price
        data[f'avg_price_last_{N}_days'] = data.groupby('SKU')['price'].transform(lambda x: x.rolling(window=N).mean())
        
        # Calculate the absolute percentage difference between the current price and the N-day average price
        data['price_variation'] = abs(data['price'] - data[f'avg_price_last_{N}_days']) / data[f'avg_price_last_{N}_days']

        # Identify rows where the price varied by less than the threshold from the N-day average price
        stability_periods = (data['price_variation'] < threshold)

        # Randomly select 98% of the stability periods to be removed
        indices_to_remove = data[stability_periods].sample(frac=removal_percentage).index
        
        # Remove the selected indices from the data
        data = data.drop(indices_to_remove)

        # Create a separate DataFrame for rows with insufficient data
        insufficient_data = data[data[f'avg_price_last_{N}_days'].isna() & data['price'].notna()]

        # log and save parameters of the function
        self.logger.info(f"filter_stability_periods: N={N}, threshold={threshold}, removal_percentage={removal_percentage}")
        # save parameters of the function
        self.params['filter_stability_periods'] = {'N': N, 'threshold': threshold, 'removal_percentage': removal_percentage}
    
        return data, insufficient_data

        





