import xgboost as xgb
import numpy as np
import wandb

class QuantileXGB:
    def __init__(self, quantiles=[0.05, 0.5, 0.95], **params):
        self._quantiles = quantiles
        self._params = params
        self._models = {}

    @property   
    def models(self):
        return self.models
        
    def fit(self, X, y, **fit_params):
        for q in self._quantiles:
            print(f'Fitting model for quantile {q}')
            self._models[q] = xgb.XGBRegressor(objective=self.quantile_loss(q), **self._params)
            self._models[q].fit(X, y, **fit_params)
        
    def predict(self, X):
        preds = []
        for q in self.quantiles:
            preds.append(self.models[q].predict(X))
        return np.vstack(preds).T
    
    def evaluate(self, y_true, y_pred):
        return np.mean(self.quantile_loss(0.5)(y_true, y_pred))
    
    def quantile_loss(self, q):
        def loss(y_true, y_pred):
            e = y_true - y_pred
            return np.maximum(q * e, (q - 1) * e)
        return loss
    
    def log_metrics(self, metrics):
        wandb.log(metrics)

    
    def train(self, X_train, y_train, X_val, y_val, hyperparameters):
        wandb.init(config=hyperparameters)
        self.fit(X_train, y_train)
        y_pred = self.predict(X_val)
        val_loss = self.evaluate(y_val, y_pred)
        self.log_metrics({'val_loss': val_loss})
        return val_loss
    
