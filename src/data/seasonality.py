import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
from scipy.fftpack import fft
import numpy as np

class SeasonalityInspector:
    def __init__(self, date_column, target_column):
        self.date_column = date_column
        self.target_column = target_column

    def reindex_dataframe(self, data):
        complete_date_range = pd.date_range(data[self.date_column].min(), data[self.date_column].max())
        data_reindexed = data.set_index(self.date_column).reindex(complete_date_range).fillna(0).reset_index().rename(columns={'index': self.date_column})
        return data_reindexed
        
    def plot_time_series(self, data):
        plt.figure(figsize=(10, 4))
        plt.plot(data[self.date_column], data[self.target_column])
        plt.title('Time Series Plot')
        plt.xlabel('Date')
        plt.ylabel(self.target_column)
        plt.show()
    
    def plot_decomposition(self, data, period):
        ts_data = data[[self.date_column, self.target_column]].set_index(self.date_column)
        result = seasonal_decompose(ts_data, period=period)
        result.plot()
        plt.show()
     
    def plot_autocorrelation(self, data, lags):
        plot_acf(data[self.target_column], lags=lags)
        plt.show()



class QuickSeasonalityInspector:
    def __init__(self):
        pass
    
    def fast_fourier_transform(self, data, target_column):
        # Apply FFT and get frequencies
        sp = np.fft.fft(data[target_column].dropna()) 
        freq = np.fft.fftfreq(len(sp))
        
        # Get dominant frequency
        dominant_frequency = np.argmax(np.abs(sp))
        # Return period (inverse of frequency)
        return abs(1 / freq[dominant_frequency]) if freq[dominant_frequency] != 0 else float('inf') 
    
    def rolling_stats(self, data, target_column, window):
        rolling_mean = data[target_column].rolling(window=window).mean()
        rolling_var = data[target_column].rolling(window=window).var()
        
        # Check if mean and variance are relatively constant
        return rolling_mean.std() < rolling_mean.mean() * 0.1, rolling_var.std() < rolling_var.mean() * 0.1




