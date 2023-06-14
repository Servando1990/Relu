import pandas as pd
from typing import List, Iterable
import holidays
import numpy as np



class FeatureEngineeringProcess:
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

        df["gmv"] = df["quantity"] * df["price"] #TODO hardcoded variables
        #compute gmv per product for the last N days
        df["gmv_last_7_days"] = df.groupby("sku")["gmv"].transform(lambda x: x.rolling(window).sum())

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
        if "holidays" in features:
            print('updated ___')
            holiday_dates = holidays.CountryHoliday('PL', years=df[date_feature].dt.year)
            df[date_feature + "_holidays"] = df[date_feature].dt.date.map(holiday_dates).notna().astype(int)

        return df
    
    def price_sales_correlation_features(self, 
                          data: pd.DataFrame, 
                          N: int, 
                          u_range: Iterable[int], 
                          v_range: Iterable[int]) -> pd.DataFrame:
        """
        Generate features for pricing model.

        :param data: DataFrame with 'price' and 'sales' columns.
        :param N: Lookback period in days.
        :param u_range: Range of values for u in correlation calculation.
        :param v_range: Range of values for v in correlation calculation.
        :return: DataFrame with generated features.
        """

        # Generate price features: average, min, max, SD for the last N days
        for i in range(N):
            data[f'price_{i}'] = data['price'].shift(i)
        data['avg_price_last_N_days'] = data[[f'price_{i}' for i in range(N)]].mean(axis=1)
        data['min_price_last_N_days'] = data[[f'price_{i}' for i in range(N)]].min(axis=1)
        data['max_price_last_N_days'] = data[[f'price_{i}' for i in range(N)]].max(axis=1)
        data['std_price_last_N_days'] = data[[f'price_{i}' for i in range(N)]].std(axis=1)

        # Generate sales features: average, min, max, SD for the last N days
        for i in range(N):
            data[f'sales_{i}'] = data['sales'].shift(i)
        data['avg_sales_last_N_days'] = data[[f'sales_{i}' for i in range(N)]].mean(axis=1)
        data['min_sales_last_N_days'] = data[[f'sales_{i}' for i in range(N)]].min(axis=1)
        data['max_sales_last_N_days'] = data[[f'sales_{i}' for i in range(N)]].max(axis=1)
        data['std_sales_last_N_days'] = data[[f'sales_{i}' for i in range(N)]].std(axis=1)

        # Generate correlation features between sales and prices for the last N days
        for u in u_range:
            for v in v_range:
                data[f'f_corr_{u}_{v}'] = np.sum(data['price']**u * data['sales']**v) / (np.sum(data['price']**u) * np.sum(data['sales']**v)) 
                #np.sum(data['price']**u * data['sales']**v) / (np.sum(data['price']**u) * np.sum(data['sales']**v))
        
        # Normalize price and sales features
        price0 = data['price'].rolling(28).mean()  # or use some other method to compute price0
        demand0 = data['sales'].rolling(28).mean()  # or use some other method to compute demand0

        data['avg_price_last_N_days_normalized'] = np.log(data['avg_price_last_N_days'] / price0)
        data['avg_sales_last_N_days_normalized'] = np.log(data['avg_sales_last_N_days'] / demand0)

        data['std_price_last_N_days_normalized'] = data['std_price_last_N_days'] / data['avg_price_last_N_days']
        data['std_sales_last_N_days_normalized'] = data['std_sales_last_N_days'] / data['avg_sales_last_N_days']

        return data
    def price_sales_correlation_features_updated(self, data: pd.DataFrame, 
                                     N: int, 
                                     u_range: Iterable[int], 
                                     v_range: Iterable[int]) -> pd.DataFrame:
    
        # Loop through each combination of u and v
        for u in u_range:
            for v in v_range:
                # Calculate the numerator part of the formula
                numerator = (data['price']**u * data['sales']**v).rolling(window=N).sum()
                
                # Calculate the sum of price to the power of u over the last N days
                sum_price_u = data['price']**u.rolling(window=N).sum()
                
                # Calculate the sum of sales to the power of v over the last N days
                sum_sales_v = data['sales']**v.rolling(window=N).sum()
                
                # Calculate the correlation feature according to the formula
                data[f'f_corr_{u}_{v}'] = numerator / (sum_price_u * sum_sales_v)
        
        return data
    
    def normalize_features(
            self, 
            data: pd.DataFrame, 
            N: int, 
            long_period: int = 28) -> pd.DataFrame:
    
        # Calculate average price and sales for the last N days
        data['avg_price_last_N_days'] = data['price'].rolling(window=N).mean()
        data['avg_sales_last_N_days'] = data['sales'].rolling(window=N).mean()
        
        # Calculate reference price and demand
        price0 = data['price'].rolling(window=long_period).mean()
        demand0 = data['sales'].rolling(window=long_period).mean()

        # Normalize average price and sales
        data['normalized_log_avg_price'] = np.log(data['avg_price_last_N_days'] / price0)
        data['normalized_log_avg_sales'] = np.log(data['avg_sales_last_N_days'] / demand0)

        # Normalize standard deviation
        data['normalized_std_price'] = data['price'].rolling(window=N).std() / data['avg_price_last_N_days']
        data['normalized_std_sales'] = data['sales'].rolling(window=N).std() / data['avg_sales_last_N_days']
        
        return data

    def filter_stability_periods(self, 
                                 data: pd.DataFrame,
                                   threshold: float = 0.05, 
                                   removal_percentage: float = 0.98) -> pd.DataFrame:
        
        # Sort the data by SKU and date
        data = data.sort_values(by=['SKU', 'date'])
        
        # Calculate the 42-day average price
        data['avg_price_last_42_days'] = data.groupby('SKU')['price'].transform(lambda x: x.rolling(window=42).mean())
        
        # Calculate the absolute percentage difference between the current price and the 42-day average price
        data['price_variation'] = abs(data['price'] - data['avg_price_last_42_days']) / data['avg_price_last_42_days']
        
        # Identify rows where the price varied by less than the threshold from the 42-day average price
        stability_periods = data['price_variation'] < threshold
        
        # Randomly select 98% of the stability periods to be removed
        indices_to_remove = data[stability_periods].sample(frac=removal_percentage).index
        
        # Remove the selected indices from the data
        data = data.drop(indices_to_remove)
        
        return data


    



    

    

