import numpy as np


# Define demand curve function
def demand_curve(price, prices, predicted_sales):
    # This function interpolates the predicted sales for a given price
    interpolated_sales = np.interp(price, prices, predicted_sales)
    #print(f'Interpolated Sales for price {price}: {interpolated_sales}')

    return interpolated_sales


# Define profit function
def profit(gmv0, r_lambda, E_r_lambda, c):
    # This function calculates the profit as described in the article
    if r_lambda is None:
        raise ValueError("r_lambda cannot be None")
    
    calculated_profit = np.sum(gmv0 * E_r_lambda * (r_lambda - c))
    #print(f'Calculated Profit: {calculated_profit}')
    return calculated_profit


# Define optimal price formula
def calculate_optimal_price(c, lambda_value, s):
    # This function calculates the optimal price using the formula given in the article
    if (lambda_value + 1) * (s - 1) == 0:
        return None
    else:
        optimal_price = c * (lambda_value * s) / ((lambda_value + 1) * (s - 1))
        #print(f'Optimal Price: {optimal_price}')

        return optimal_price
    


# Pricing algorithm function
def pricing_algorithm(prices, predicted_sales, c, s, initial_lambda=5, delta_lambda=0.1, P0=0, max_iterations=1000, D_lambda=4):
    # Step 0: Select initial λ = 5.
    lambda_value = initial_lambda
    iteration = 0

    #print(f'Initial Lambda Value: {lambda_value}')

    # Loop until profit constraint is satisfied or max iterations are reached
    while True:
        # Step 2: For current λ calculate prices, and find out what Profit(λ) is.
        # Calculate prices with lambda
        prices_with_lambda = calculate_optimal_price(c, lambda_value, s)

        # Use demand_curve function to calculate E_r_lambda
        E_r_lambda = demand_curve(prices_with_lambda, prices, predicted_sales)
        gmv0 = demand_curve(prices, prices, predicted_sales)

        # Calculate the Profit(λ)
        current_profit = profit(gmv0, prices_with_lambda, E_r_lambda, c)

        # Step 3: If Profit(λ) < P0, slightly increase λ. If Profit(λ) > P0, slightly decrease λ.
        if current_profit < P0:
            lambda_value += delta_lambda
            #print(f'Current Profit < P0; Increasing Lambda to {lambda_value}')
        else:
            lambda_value -= delta_lambda
            #print(f'Current Profit >= P0; Decreasing Lambda to {lambda_value}')

        # Limit the change in λ in a week to D_lambda percent.
        lambda_value = max(lambda_value, initial_lambda * (1 - D_lambda / 100))
        lambda_value = min(lambda_value, initial_lambda * (1 + D_lambda / 100))

        # Increment the iteration
        iteration += 1

        #print(f'Iteration: {iteration}, Lambda Value: {lambda_value}, Current Profit: {current_profit}')

        # Check the exit condition
        if current_profit >= P0 or iteration >= max_iterations:
            break

    # Result of pricing algorithm
    print(f'Final Optimal Price: {prices_with_lambda}, Final Lambda Value: {lambda_value}')
    return prices_with_lambda, lambda_value

def grid_search_lambda(P0, prices, predicted_sales, c, s, lambda_min=3, lambda_max=5, lambda_step=1, D_lambda=4):
    # Iterate over a range of lambda values
    lambda_values = range(lambda_min, lambda_max + 1, lambda_step)
    best_prices = []
    best_lambdas = []

    #print(f'Starting grid search with Lambda values: {list(lambda_values)}')

    # Loop through each lambda value
    for lambda_multiplier in lambda_values:

        print(f'\nProcessing Lambda Multiplier: {lambda_multiplier}')
        # Call pricing_algorithm function to calculate optimal price and lambda for the current lambda_multiplier
        optimal_price, optimal_lambda = pricing_algorithm(prices, predicted_sales, c, s, initial_lambda=lambda_multiplier, P0=P0, D_lambda=D_lambda)

        print(f'Optimal Price for Lambda Multiplier {lambda_multiplier}: {optimal_price}')
        print(f'Optimal Lambda Value for Lambda Multiplier {lambda_multiplier}: {optimal_lambda}')

        # Store the optimal price and lambda
        best_prices.append(optimal_price)
        best_lambdas.append(optimal_lambda)

    # Return the best prices and corresponding lambda values
    print(f'\nFinal Results - Best Prices: {best_prices}, Best Lambda Values: {best_lambdas}')
    return best_prices, best_lambdas



