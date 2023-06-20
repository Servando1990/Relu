
import modal
import pandas as pd
import time

xgb_image = modal.Image.debian_slim().pip_install("pandas==1.4.2", "xgboost", "scikit-learn")
stub = modal.Stub("example-xgb-started")

if stub.is_inside():
    from xgboost import XGBRegressor
    import pandas as pd


@stub.function(image=xgb_image)
def perform_inference(train_data, test_data):
    # Fit the XGBoost model
    xgboost_model = XGBRegressor()
    xgboost_model.fit(train_data.drop(columns=['SKU', 'sales']), train_data['sales'])

    # Group test data by SKU
    grouped = test_data.groupby('SKU')

    # Create an empty dataframe to store the results
    results = pd.DataFrame(columns=['SKU', 'price', 'predicted_sales_xgboost'])

    # Perform inference for each SKU in test data
    for sku, group_data in grouped:
        print(f"Performing inference for SKU: {sku}")

        # Select one row from group data (one set of feature values for this SKU)
        feature_values = group_data.iloc[0]
        #print(f"Feature values: {feature_values.shape}")

        # Set the base price from data
        base_price = feature_values['price']

        # Define the price points
        price_points = [base_price, 
                        base_price * 0.9,   # base_price - 10%
                        base_price * 0.85,  # base_price - 15%
                        base_price * 1.1,   # base_price + 10%
                        base_price * 1.15]  # base_price + 15%

        # For each price point, use the trained model to predict sales and print the result
        for price in price_points:
            # Change the price while keeping other features constant
            test_features = feature_values.copy()
            test_features['price'] = price

            # Remove the SKU field to match the input shape expected by the model
            test_features = test_features.drop(['SKU', 'sales'])
            #print(f"Test features: {test_features.shape}")
 
            # Start the timer
            start_time = time.time()

            # Predict sales using the XGBoost model and print the result
            predicted_sales_xgboost = xgboost_model.predict(test_features.values.reshape(1, -1))[0]

            # Stop the timer
            elapsed_time = time.time() - start_time
            print(f"Price: {price}, XGBoost Predicted Sales: {predicted_sales_xgboost}, Elapsed Time: {elapsed_time:.4f} seconds")

            # Append the results to the dataframe
            results = results.concat({'SKU': sku, 'price': price, 'predicted_sales_xgboost': predicted_sales_xgboost}, ignore_index=True)



        print("\n")

    return results
    
@stub.local_entrypoint()
def main():
    # load train.pickle and test.pickle from ..data/processed folder
    train_data = pd.read_pickle('/Users/servandodavidtorresgarcia/Servando/Relu/Relu/data/processed/train.pkl')
    
    test_data = pd.read_pickle('/Users/servandodavidtorresgarcia/Servando/Relu/Relu/data/processed/test.pkl')
    
    # Perform inference using the trained model
    
    df = perform_inference.call(train_data, test_data)
    df.to_csv('/Users/servandodavidtorresgarcia/Servando/Relu/Relu/data/processed/predictions.csv', index=False)

