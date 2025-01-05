import numpy as np
from processes_predictors.predict import predict

def parallel_cv(params):
    model, base_model, X_train, Y_train, tscv, use_base_model_fore, raw_Y, model_params = params
    scores = []
    for train_idx, test_idx in tscv.split(X_train):
        X_train_cv, X_test_cv = X_train[train_idx], X_train[test_idx]
        y_train_cv, y_test_cv = Y_train[train_idx], Y_train[test_idx]

        if use_base_model_fore:    
            naive_forecast_cv, _, _, naive_forecasts_cv = predict(base_model, X_train_cv, y_train_cv, X_test_cv)
        else:
            naive_forecast_cv = raw_Y[test_idx]
            naive_forecasts_cv = raw_Y[train_idx]
        # Train kernel regression model
        if model == 'SVR':
            pr_cv, _, _, _ = predict('SVR', X_train_cv, y_train_cv, X_test_cv, q_kernel=model_params['q_kernel'], q_data=model_params['q_data'], epsilon=model_params['epsilon'], C=model_params['C'], insample_mae=False)
        elif model == "cSVR":
            pr_cv, _, _, _ = predict('cSVR', X_train_cv, y_train_cv, X_test_cv, q_kernel=model_params['q_kernel'], q_data=model_params['q_data'], naive_forecasts=naive_forecasts_cv, naive_forecast=naive_forecast_cv, q_kernel_naive=model_params['q_kernel_naive'], q_data_naive=model_params['q_data_naive'], epsilon=model_params['epsilon'], C=model_params['C'], insample_mae=False)
        elif model == "ccSVR":
            pr_cv, _, _, _ = predict('ccSVR', X_train_cv, y_train_cv, X_test_cv, q_kernel=model_params['q_kernel'], q_data=model_params['q_data'], naive_forecasts=naive_forecasts_cv, naive_forecast=naive_forecast_cv, q_kernel_naive=model_params['q_kernel_naive'], q_data_naive=model_params['q_data_naive'], q_kernel_div=model_params['q_kernel_div'], q_data_div=model_params['q_data_div'], div_impact=model_params['div_impact'], epsilon=model_params['epsilon'], C=model_params['C'], insample_mae=False)
        scores.append(np.abs(y_test_cv[0] - pr_cv))

    return np.mean(scores)
