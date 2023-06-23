import modal
import time
import pandas as pd
from src.pricing.pricing_algorithm import grid_search_lambda
import numpy as np

xgb_image_pricing = modal.Image.debian_slim().pip_install("pandas==1.4.2")
stub = modal.Stub("example-pricing_algorithm")

if stub.is_inside():
    import pandas as pd
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')


@stub.function(image=xgb_image_pricing)
def find_optimal_price(sku_row, c=10, s=2, P0=100):

    print(f'sku_row content: {sku_row}')
    # Extracting SKU, price and predicted sales
    sku = sku_row['SKU']
    price = sku_row['price']
    predicted_sale = sku_row['predicted_sales_xgboost']
    
    print(f'\nProcessing SKU: {sku}')
    print(f'Original Price: {price}, Predicted Sales: {predicted_sale}')
    
    # Call grid_search_lambda function to find optimal price for the current SKU
    prices = np.array([price])
    predicted_sales = np.array([predicted_sale])

    # start the timer
    start_time = time.time()
    
    print(f'Starting grid search for SKU: {sku}')
    best_prices, best_lambdas = grid_search_lambda(P0, prices, predicted_sales, c, s)

    # stop the timer
    elapsed_time = time.time() - start_time
    
    # Extract the optimal price
    optimal_price = best_prices[0]
    
    print(f'Optimal Price for SKU {sku}: {optimal_price}, Elapsed Time: {elapsed_time:.4f} seconds\n')

    
    # Return the optimal price
    return optimal_price

@stub.local_entrypoint()
def main():

    # start the timer
    start_time = time.time()
    predicted_sales = pd.read_csv('/Users/servandodavidtorresgarcia/Servando/Relu/Relu/data/processed/predictions.csv')

    optimal_prices = predicted_sales.apply(find_optimal_price) 
    result_df = pd.DataFrame({'SKU': predicted_sales['SKU'], 'optimal_prices': optimal_prices})
    result_df.to_csv('/Users/servandodavidtorresgarcia/Servando/Relu/Relu/data/processed/optimal_prices.csv', index=False)
    # stop the timer
    elapsed_time = time.time() - start_time
    print(f"Total Elapsed Time for find_optimal_price: {elapsed_time:.4f} seconds")
