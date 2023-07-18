import numpy as np

# Define the QueryRMSE metric
def query_rmse(y_true, y_pred, groups):
    group_corrections = {}
    for group in np.unique(groups):
        idx = np.where(groups == group)
        y_true_group = y_true[idx]
        y_pred_group = y_pred[idx]
        correction = np.mean(y_true_group - y_pred_group)
        group_corrections[group] = correction
    
    corrected_preds = y_pred + np.array([group_corrections[group] for group in groups])
    return np.sqrt(np.mean((y_true - corrected_preds)**2))