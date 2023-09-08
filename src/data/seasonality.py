import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
from scipy.fftpack import fft
import numpy as np

class SeasonalityInspector:
    def __init__(self):
        pass
    
    def plot_time_series(self, data, date_column, target_column):
        plt.figure(figsize=(10, 4))
        plt.plot(data[date_column], data[target_column])
        plt.title('Time Series Plot')
        plt.xlabel('Date')
        plt.ylabel(target_column)
        plt.show()
    
    def plot_decomposition(self, data, date_column, target_column, period):
        ts_data = data[[date_column, target_column]].set_index(date_column)
        result = seasonal_decompose(ts_data, period=period)
        result.plot()
        plt.show()
    
    def plot_autocorrelation(self, data, target_column, lags=40):
        plot_acf(data[target_column], lags=lags)
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




