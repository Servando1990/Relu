import numpy as np

# Define demand curve function
def demand_curve(price, prices, predicted_sales):
    return np.interp(price, prices, predicted_sales)

# Define profit function
def profit(gmv0, r_lambda, E_r_lambda, c, P0):
    return np.sum(gmv0 * E_r_lambda * (r_lambda - c)) - P0


# Run the pricing algorithm to find the optimal value of λ

def pricing_algorithm(P0, demand_curve, prices, predicted_sales, c, initial_lambda=5, delta_lambda=4):
    # Step 0: Select an initial value of λ = 5. Select fixed global stability parameter D_\lambda = 4.
    lambda_value = initial_lambda

    while True:

        # Calculate prices with lambda
        prices_with_lambda = prices * lambda_value

        # Step 2: For current λ calculate prices, and find out what a
        gmv0 = demand_curve(P0, prices, predicted_sales)
        # Use demand_curve function to calculate E_r_lambda
        E_r_lambda = demand_curve(prices_with_lambda, prices, predicted_sales)
        r_lambda = E_r_lambda + (gmv0 - E_r_lambda) / (1 + np.exp((prices - prices_with_lambda) / c))


        current_profit = profit(gmv0, r_lambda, E_r_lambda, c, P0)
        # Step 3: If \mathrm{Profit}(\lambda)< P_0, slightly increase λ.
        # If \mathrm{Profit}(\lambda)> P_0, slightly decrease λ.
        # Change in λ in a week should be no more than D_\lambdapercent.
        if current_profit < 0:
            lambda_value += delta_lambda
        elif current_profit > 0:
            lambda_value -= delta_lambda
        else:
            break

        # Step 1: Select/update value of P0 according to business needs and strategy.
        # Go to Step 2.

    # Return the optimal value of λ
    return lambda_value

# Define grid search lambda function
def grid_search_lambda(P0, demand_curve, prices, predicted_sales, c, lambda_min=3, lambda_max=10, lambda_step=1):
    # Step 0: Select initial λ = 5. Select fixed global stability parameter D_\lambda = 4.
    lambda_values = range(lambda_min, lambda_max + 1, lambda_step)
    best_prices = []

    for lambda_multiplier in lambda_values:
        # Step 2: For current λ calculate prices, and find out what a \mathrm{Profit}(\lambda)is.
        lambda_value = pricing_algorithm(P0, demand_curve, prices, predicted_sales, c, lambda_multiplier)
        prices_with_lambda = prices * lambda_value
        profits = profit(demand_curve(P0, prices_with_lambda, predicted_sales), 1, 1, c, P0)
        best_price = prices_with_lambda[np.argmax(profits)]
        best_prices.append(best_price)

        # Step 1: Select/update value of P0 according to business needs and strategy.
        # Go to Step 2.

    # Return the best prices
    return best_prices
