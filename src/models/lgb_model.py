
import numpy as np
import wandb
from lightgbm import LGBMRegressor
import wandb
from wandb.lightgbm import wandb_callback
import numpy as np

class QuantileLGBMColab:
    def __init__(self,
                 quantiles=[0.1, 0.5, 0.9],
                 early_stopping_rounds=20,
                 **params):
        self._quantiles = quantiles
        self._early_stopping_rounds = early_stopping_rounds
        self._params = params
        self._models = {}

    def fit(self, X, y, X_val, y_val, **fit_params):
        for q in self._quantiles:
            print(f'Fitting model for quantile {q}')
            self._models[q] = LGBMRegressor(objective='quantile', alpha=q,
                                            **self._params,
                                            early_stopping_rounds=self._early_stopping_rounds)
            self._models[q].fit(X, y,
                                eval_set=[(X_val, y_val)],
                                callbacks=[wandb_callback()],
                                **fit_params)
            # Log feature importances with wandb
            wandb.log({'feature_importances': self._models[q].feature_importances_})

    def predict(self, X):
        preds = []
        for q in self._quantiles:
            preds.append(self._models[q].predict(X))
        return np.vstack(preds).T

    def evaluate(self, y_true, y_pred):
        # You can define any evaluation metric here
        # For instance, mean absolute error for each quantile
        y_true_repeated = np.repeat(y_true[:, np.newaxis], y_pred.shape[1], axis=1)
        
        return np.mean(np.abs(y_true_repeated - y_pred), axis=0)

    def log_metrics(self, metrics):
        wandb.log(metrics)

    def train(self, X_train, y_train, X_val, y_val, hyperparameters):
        wandb.init(config=hyperparameters)
        self.fit(X_train, y_train, X_val, y_val)
        y_pred = self.predict(X_val)
        val_loss = self.evaluate(y_val, y_pred)
        self.log_metrics({'val_loss': val_loss})
        return val_loss

    # export models for later inference
    def export(self, path):
        for q in self._quantiles:
            self._models[q].booster_.save_model(f'{path}/model_{q}.txt')





    
