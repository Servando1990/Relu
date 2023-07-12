import numpy as np
from typing import List

def demand_curve(price: np.array, prices: np.array, predicted_sales: np.array) -> np.array:
    return np.interp(price, prices, predicted_sales)

def total_profit(GMV: np.array, r_lambda: np.array, E_r_lambda: np.array, c: np.array) -> np.array:
    if r_lambda is None:
        raise ValueError("r_lambda cannot be None")
    return GMV * E_r_lambda * (r_lambda - c)

def calculate_optimal_price(c: np.array, lambda_value: np.array, s: np.array) -> np.array:
    if (lambda_value + 1) * (s - 1) == 0:
        return None
    else:
        return c * (lambda_value * s) / ((lambda_value + 1) * (s - 1))

def calculate_GMV(prices: np.array, predicted_sales: np.array) -> np.array:
    return np.sum(prices * predicted_sales)

def binary_search(prices: np.array, predicted_sales: np.array, c: np.array, s: np.array, P0=0, left_lambda=3, right_lambda=5, D_lambda=4, max_iterations=1000) -> List[np.array]:
    for iteration in range(max_iterations):
        lambda_value = (right_lambda + left_lambda) / 2

        prices_with_lambda = calculate_optimal_price(c, lambda_value, s)

        E_r_lambda = demand_curve(prices_with_lambda, prices, predicted_sales)
        GMV = calculate_GMV(prices, predicted_sales)

        current_profit = total_profit(GMV, prices_with_lambda, E_r_lambda, c)

        if abs(np.sum(current_profit) - P0) < 1e-9:  # tolerance level
            return prices_with_lambda, lambda_value

        if np.sum(current_profit) < P0:
            left_lambda = lambda_value
        else:
            right_lambda = lambda_value

    return prices_with_lambda, lambda_value

def grid_search_lambda(P0: float, prices: np.array, predicted_sales: np.array, c: np.array, s: np.array, lambda_min=3, lambda_max=5, lambda_step=0.1, D_lambda=4) -> List[np.array]:
    # Determine initial bounds
    lambda_values = np.arange(lambda_min, lambda_max + lambda_step, lambda_step)
    best_lambda = None
    best_profit = float('-inf')

    # Iterate over possible lambda values
    for lambda_value in lambda_values:
        # Calculate optimal price using binary search
        optimal_price, optimal_lambda = binary_search(prices, predicted_sales, c, s, P0=P0, left_lambda=lambda_value, right_lambda=lambda_value+lambda_step, D_lambda=D_lambda)
        GMV = calculate_GMV(prices, predicted_sales)
        E_r_lambda = demand_curve(optimal_price, prices, predicted_sales)
        current_profit = total_profit(GMV, optimal_price, E_r_lambda, c)

        # Update best lambda value if this is the maximum profit so far
        if np.sum(current_profit) > best_profit:
            best_profit = np.sum(current_profit)
            best_lambda = optimal_lambda

    # Return optimal price with best lambda
    optimal_price = calculate_optimal_price(c, best_lambda, s)

    return optimal_price, best_lambda



