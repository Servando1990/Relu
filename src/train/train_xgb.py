
import modal
import pandas as pd
import time
import numpy as np
import wandb
#TODO this script doenst work, it needs to be fixed
xgb_wnb = modal.Image.debian_slim().pip_install("pandas==1.4.2", "xgboost", "scikit-learn", "wandb")
stub = modal.Stub("xgb_weight_and_biases")

if stub.is_inside():
    import pandas as pd
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')


#@stub.function(mounts=[modal.Mount.from_local_dir("~/foo", remote_path="/root/foo")])
sweep_config = {
    'method': 'bayes',  # We are using Bayesian optimization
    'metric': {
    'name': 'val_loss',  # We want to minimize validation loss
    'goal': 'minimize'  
    },
    'parameters': {
        'learning_rate': {'min': 0.01, 'max': 0.1},
        'max_depth': {'min': 3, 'max': 10},
        'n_estimators': {'min': 50, 'max': 200},
        'subsample': {'min': 0.5, 'max': 1.0},
        'colsample_bytree': {'min': 0.5, 'max': 1.0},
        'reg_alpha': {'min': 0.0, 'max': 1.0},
        'reg_lambda': {'min': 0.0, 'max': 1.0},
    }
    }

@stub.function(image=xgb_wnb)
def sweep_run():
    from src.models.lgb_model import QuantileXGB
    X_train = pd.read_pickle('/Users/servandodavidtorresgarcia/Servando/Relu/Relu/data/processed/X_train.pkl')
    y_train = pd.read_pickle('/Users/servandodavidtorresgarcia/Servando/Relu/Relu/data/processed/y_train.pkl')
    X_val = pd.read_pickle('/Users/servandodavidtorresgarcia/Servando/Relu/Relu/data/processed/X_val.pkl')
    y_val = pd.read_pickle('/Users/servandodavidtorresgarcia/Servando/Relu/Relu/data/processed/y_val.pkl')
    #quantile_xgb = QuantileXGB()

    with wandb.init() as run:
        config = wandb.config
        model = QuantileXGB(**config)
        val_loss = model.train(X_train, y_train, X_val, y_val, config)
        wandb.log({'val_loss': val_loss})


    
@stub.local_entrypoint()
def main():

    # start the timer
    start_time = time.time()
    sweep_id = wandb.sweep(sweep_config, project='relu-project')
    wandb.agent(sweep_id, function=sweep_run)

    elapsed_time = time.time() - start_time
    print(f"Total Elapsed Time for training: {elapsed_time:.4f} seconds")



