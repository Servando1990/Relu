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

            for q in self._quantiles:
                with mlflow.start_run():
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
                    for idx, importance in enumerate(model.feature_importances_):
                        mlflow.log_metric(f'quantile_{q}_feature_importance_{idx}', importance)



    def predict(self, X_val, y_val):
        preds = []
        for q in self._quantiles:

            pred = self._models[q].predict(X_val)
            preds.append(pred)

            #Log individual loss for this quantile
            val_loss_q = np.mean(np.abs(y_val - pred))
            mlflow.log_metric(f'val_loss_quantile_{q}', val_loss_q)

        return np.vstack(preds).T

    def evaluate(self, y_true, y_pred):
        y_true_repeated = np.repeat(y_true[:, np.newaxis], y_pred.shape[1], axis=1)
        return np.mean(np.abs(y_true_repeated - y_pred), axis=0)

    def optimize(self, objective, n_trials=100):

        print(f'Optimizing hyperparameters')    
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params

    def register_model(self, model_name, model_version):
        mlflow.register_model(model_name, model_version)

    # export models for later inference
    def export(self, path):
        for q in self._quantiles:
            self._models[q].booster_.save_model(f'{path}/model_{q}.txt')






    
