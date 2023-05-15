import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(42)

# Number of data points
n = 1000

# Dates ranging over a year
dates = pd.date_range(start='2022-01-01', periods=n)

# Product categories
categories = ['Electronics', 'Furniture', 'Grocery', 'Clothing']
product_categories = np.random.choice(categories, n)

# Products
products = [f'Product_{i}' for i in range(1, 51)]  # 50 products
product_list = np.random.choice(products, n)

# Prices normally distributed and positive
prices = np.abs(np.random.normal(loc=100, scale=20, size=n))

# Quantity sold - we assume that it's a Poisson distribution
quantity = np.random.poisson(lam=10, size=n)

# Competitor's price - usually it would be around the actual product price
competitor_prices = prices + np.random.normal(loc=0, scale=10, size=n)

# Discounts
discounts = np.random.choice([0, 5, 10, 15, 20], size=n)  # in percentage

# Create a dataframe
df = pd.DataFrame({
    'date': dates,
    'product': product_list,
    'category': product_categories,
    'price': prices,
    'quantity': quantity,
    'competitor_price': competitor_prices,
    'discount': discounts
})

# Black Friday dates
df['black_friday_date'] = pd.to_datetime(['2022-11-25' if date.month >= 11 else '2023-11-24' for date in df['date']])

df.to_csv('synthetic_data.csv', index=False)
