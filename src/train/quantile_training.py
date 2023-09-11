from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# import Dict
from typing import Dict


class DemandCurveTrainer:
    def __init__(self, X_train, y_train, X_val, y_val, 
                 base_params = None, quantile_alphas=[0.1, 0.5, 0.9], 
                 tune_params=False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.base_params = base_params if base_params else {
            'n_jobs': 1,
            'max_depth': 5,
            'min_data_in_leaf': 10,
            'subsample': 0.8,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'colsample_bytree': 0.8,
            'boosting_type': 'gbdt'
        }
        self.quantile_alphas = quantile_alphas
        self.models = {}
        self.tune_params = tune_params
        if self.tune_params:
            self.tuned_params = self.hyperparameter_tuning()
        else:
            self.tuned_params = base_params
    
    def objective(self, trial):
        params = {
            'n_jobs': 1,
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'boosting_type': 'gbdt'
        }
        model = LGBMRegressor(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)
        return mean_squared_error(self.y_val, y_pred)
    
    def hyperparameter_tuning(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=50)
        return study.best_params
    
    def train_models(self):
        for q in self.quantile_alphas:
            print(f'Fitting model for quantile {q}')
            params = {**self.base_params, **{'objective': 'quantile', 'alpha': q}}
            model = LGBMRegressor(**params)
            model.fit(self.X_train, self.y_train,
                      eval_set=[(self.X_val, self.y_val)],
                      early_stopping_rounds=20)
            self.models[q] = model
    
    def evaluate_models(self):
        for q in self.quantile_alphas:
            model = self.models[q]
            y_train_pred = model.predict(self.X_train)
            y_val_pred = model.predict(self.X_val)
            loss_train = self.quantile_loss(self.y_train, y_train_pred, q)
            loss_val = self.quantile_loss(self.y_val, y_val_pred, q)
            print(f"Quantile {q}: Training Loss = {loss_train.mean()}, Validation Loss = {loss_val.mean()}")
    
    def quantile_loss(self, y_true, y_pred, quantile_alpha):
        e = y_true - y_pred
        return np.maximum(quantile_alpha * e, (quantile_alpha - 1) * e)
    
    def predict_demand_curve(self, test_data: pd.DataFrame, sku_column: str, base_price_feature: str) -> Dict:
        sku_demand_curves = {}
        price_points_per_sku = {}
        demand_curve_pricing = {}
        
        price_multipliers = [0.85, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10, 1.15, 1.20]
        
        grouped_test_data = test_data.groupby(sku_column)
        
        for sku, group_data in grouped_test_data:
            sample_row = group_data.iloc[0].copy()
            
            if sku_column in sample_row:
                sample_row.drop(sku_column, inplace=True)
            
            base_price = sample_row[base_price_feature]
            price_points = [base_price * factor for factor in price_multipliers]
            price_points_per_sku[sku] = price_points
            
            demand_predictions = {}
            for q, model in self.models.items():
                demand_predictions[q] = []
                for price in price_points:
                    sample_row[base_price_feature] = price
                    prediction = model.predict([sample_row])
                    demand_predictions[q].append(prediction[0])
                    
            demand_curve = {q: list(zip(price_points, demand_predictions[q])) for q in self.quantile_alphas}
            sku_demand_curves[sku] = demand_predictions
            demand_curve_pricing[sku] = demand_curve

        return sku_demand_curves, demand_curve_pricing, price_points_per_sku

    
    # FIXME: This function is not working
    def plot_feature_importance(self):
        for q in self.quantile_alphas:
            plt.figure()
            plt.title(f'Feature importances for quantile {q}')
            sorted_idx = self.models[q].feature_importances_.argsort()
            plt.barh(self.X_train.columns[sorted_idx], self.models[q].feature_importances_[sorted_idx])
            plt.xlabel("LGBM Feature Importance")
            plt.show()
