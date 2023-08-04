import numpy as np
from typing import List, Tuple, Dict


class PricingOptimizer:
    def __init__(self, 
                 prices_base: np.ndarray, 
                 predicted_demands_base: np.ndarray,
                   costs: np.ndarray)-> None:
        """
        Initializes the PricingOptimizer with base prices, predicted demands, and costs.
        """
        self.prices_base = prices_base
        self.predicted_demands_base = predicted_demands_base
        self.costs = costs

    def L(self,
           price: float,
             demand: float,
               cost: float,
                 lambda_value: float)-> float:
        """
        The Lagrange method that calculates the value for a given price, demand, cost, and lambda.
        """
        return demand * (price + lambda_value * (price - cost))

    def total_profit(self,
                      prices: np.ndarray,
                        demands: np.ndarray,
                          costs: np.ndarray)-> float:
        """
        Calculates the total profit given prices, demands, and costs.
        """
        return np.sum((prices - costs) * demands)

    def demand_price_and_demand(self,
                                 lambda_value: float,
                                 price_multipliers: List[float] = [0.85, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10, 1.15, 1.20])-> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the optimal prices and demands for a given lambda using the Lagrange method.
        """
        prices_opt = []
        demands_opt = []
        for (sku_prices_base, sku_pr_demands_base, sku_cost) in zip(self.prices_base, self.predicted_demands_base, self.costs):
            l_max = -1000000000.0
            sku_price_opt = sku_prices_base[2]
            sku_demand_opt = 0
            sku_prices = np.array(price_multipliers) * sku_prices_base[2]
            for sku_price in sku_prices:
                sku_demand = np.interp(sku_price, sku_prices_base, sku_pr_demands_base)
                l = self.L(sku_price, sku_demand, sku_cost, lambda_value)
                if l > l_max:
                    l_max = l
                    sku_price_opt = sku_price
                    sku_demand_opt = sku_demand
            prices_opt.append(sku_price_opt)
            demands_opt.append(sku_demand_opt)

        return np.array(demands_opt), np.array(prices_opt)

    def binary_search(self,
                       P0: float,
                         left_lambda:float =3,
                           right_lambda: float =5,
                               max_iterations: int =1000,
                                 lambda_accuracy: float =0.2):
        """
        Performs a binary search to find the optimal lambda value that maximizes profit.
        """
        iterations = 0
        while right_lambda - left_lambda > lambda_accuracy and iterations < max_iterations:
            lambda_value = (right_lambda + left_lambda) / 2.0
            prices, demands = self.demand_price_and_demand(lambda_value)
            current_profit = self.total_profit(prices, demands, self.costs)

            if current_profit < P0:
                left_lambda = lambda_value
            else:
                right_lambda = lambda_value
            iterations += 1

        return prices, lambda_value
    



""" def demand_curve(price: np.array, prices: np.array, predicted_sales: np.array) -> np.array:
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

    return optimal_price, best_lambda """

# Artem approach: --------------------------------------------------------------------------------

""" def total_profit(prices, demands, costs):
    return np.sum( (prices - costs) * demands )

def demand_price_and_demand(
    lambda_value: float, 
    prices_base: np.array, #  dim (skus_count, 5) Predictions for machien learning model
    pr_demands_base: np.array #  dim (skus_count, 5)
):
    # ....
    prices_opt = prices_base[:, 2]
    # repeat vector prices_opt 5 times: np.array([prices_opt * 0.85, prices_opt * 0.9, ...])
    price_versions = prices_opt[:, None] //...// np.array([0.85, 0.9, 1.0, 1.1, 1.2])
    # TODO: choose one best version prices_opt for each sku 
    # demand_versions = interpolate(prices_base, pr_demands_base, price_opt)  # dim (skus_count, 5)
    # L = L(demand_versions, prices_versions, costs)
    # indexes = arg_max(L, dim=-1)
    # price_opt = price_versions[ indexes ]
    price_versions = prices_opt[:, None] #... np.array([0.85, 0.9, 1.0, 1.1, 1.2])
    
    # TODO: choose one best version prices_opt for each sku 
    # price_opt
    price_versions = prices_opt[:, None] #... np.array([0.95, 0.98, 1.0, 1.02, 1.05])
    # TODO: choose one best version prices_opt for each sku 
    # price_opt
    # demand_opt = interpolate(prices_base, pr_demands_base, price_opt)

    prices_opt
    for (sku_prices_base, sku_pr_demands_base, sku_cost) in zip(prices_base, pr_demands_base, costs):
        l_max = -1000000000.0
        sku_price_opt = sku_prices_base[2]
        sku_demand_opt = 0
        sku_prices = np.array([0.85, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10, 1.15, 1.20]) * sku_prices_base[2]
        for sku_price in sku_prices:
            sku_demand = np.interp(sku_prices_base, sku_pr_demands_base, )
            l = L(sku_price, sku_demand, sku_cost, lambda_value)
            if l > l_max:
                l_max = l 
                sku_price_opt = sku_price
                sku_demand_opt = sku_demand
        prices_opt.append(sku_price_opt)
        demands_opt.append(sku_demand_opt)

    return demands_opt, prices_opt


def L(price, demand, cost, lambda_value):
    return demand * (price + lambda_value * (price - cost))

def binary_search(
    prices_base: np.array, 
    predicted_sales_base: np.array,
    costs: np.array, 
    left_lambda=3, 
    right_lambda=5, 
    D_lambda=4, 
    max_iterations=1000,
    lambda_accuracy = 0.2
) -> List[np.array]:
    while right_lambda - left_lambda > lambda_accuracy:
        lambda_value = (right_lambda + left_lambda) / 2.0
        prices, demands = demand_price_and_demand(lambda_value, prices_base, predicted_sales_base)

        current_profit = total_profit(prices, demands, costs)

        if current_profit < P0:
            left_lambda = lambda_value
        else:
            right_lambda = lambda_value

    return prices, lambda_value """



