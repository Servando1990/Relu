from typing import Any
import numpy as np
import pandas as pd
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns


class Eda:
    def missing_values_table(self, df: pd.DataFrame):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * mis_val / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table.columns = ["Missing Values", "% of Total Values"]
        mis_val_table = (
            mis_val_table[mis_val_table["% of Total Values"] != 0]
            .sort_values("% of Total Values", ascending=False)
            .round(1)
        )
        print(
            f"The selected dataframe has {df.shape[1]} columns and {mis_val_table.shape[0]} columns with missing values."
        )
        return mis_val_table

    def detect_outliers_boxplot(df: pd.DataFrame, features: List[str], fac=1.5):
        if isinstance(features, str):
            features = [features]

        q1 = df.loc[:, features].quantile(0.25)
        q3 = df.loc[:, features].quantile(0.75)
        iqr = q3 - q1
        lower_threshold = q1 - fac * iqr
        upper_threshold = q3 + fac * iqr

        outliers = (
            (df[features] < lower_threshold) | (df[features] > upper_threshold)
        ).any(axis=1)
        return outliers
    
    # Show statistics
    df.groupby('cohort').describe().T.style.background_gradient(cmap='coolwarm')

    # Display correlations
    display(df.corr().style.background_gradient(cmap='coolwarm'))

    # Display variance influence factor



