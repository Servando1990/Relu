import pandas as pd
from typing import List
import holidays




class FeatureEngineeringProcess:
    def grouped_feature_eng(
        self,
        df: pd.DataFrame,
        group_features: list,
        features: list,
        target_feature: str,
    ):
        """Transform aggregation based on grouping a set of features

        Args:
            df (pd.DataFrame): Dataframe to be transformed
            group_features (list): List of features selected to group the transformation.
            features (list): List of features to be transformed.
            target_feature (str): Feature to be used in the aggregated transformation (Numerical)

        Returns:
            df: Transformed pd.DataFrame
        """

        for feature in features:
            df[feature + "_sum"] = df.groupby(group_features)[target_feature].transform(
                "sum"
            )
            df[feature + "_mean"] = df.groupby(group_features)[
                target_feature
            ].transform("mean")
            df[feature + "_min"] = df.groupby(group_features)[target_feature].transform(
                "min"
            )
            df[feature + "_max"] = df.groupby(group_features)[target_feature].transform(
                "max"
            )
            df[feature + "_count"] = df.groupby(group_features)[
                target_feature
            ].transform("count")

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

    

    

