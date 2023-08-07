import mlflow
import mlflow.lightgbm
import optuna
from lightgbm import LGBMRegressor
import numpy as np

class QuantileLGBM:
    def __init__(self,
                 quantiles=[0.1, 0.5, 0.9],
                 early_stopping_rounds=20,
                 experiment_name="QuantileLGBM"):
        self._quantiles = quantiles
        self._early_stopping_rounds = early_stopping_rounds
        self._experiment_name = experiment_name
        self._models = {}

        mlflow.set_experiment(self._experiment_name)

    def fit(self, X, y, X_val, y_val, params):
        with mlflow.start_run() as run:
            for q in self._quantiles:
                print(f'Fitting model for quantile {q}')
                model = LGBMRegressor(objective='quantile', alpha=q,
                                      **params,
                                      early_stopping_rounds=self._early_stopping_rounds)
                model.fit(X, y,
                          eval_set=[(X_val, y_val)],
                          verbose=10)
                self._models[q] = model

                # Log metrics and feature importances with MLflow
                mlflow.log_params(params)
                mlflow.log_metric(f'quantile_{q}_feature_importances', model.feature_importances_)

    def predict(self, X):
        preds = []
        for q in self._quantiles:
            preds.append(self._models[q].predict(X))
        return np.vstack(preds).T

    def evaluate(self, y_true, y_pred):
        y_true_repeated = np.repeat(y_true[:, np.newaxis], y_pred.shape[1], axis=1)
        return np.mean(np.abs(y_true_repeated - y_pred), axis=0)

    def optimize(self, objective, n_trials=100):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def register_model(self, model_name, model_version):
        mlflow.register_model(model_name, model_version)

    # export models for later inference
    def export(self, path):
        for q in self._quantiles:
            self._models[q].booster_.save_model(f'{path}/model_{q}.txt')






    
