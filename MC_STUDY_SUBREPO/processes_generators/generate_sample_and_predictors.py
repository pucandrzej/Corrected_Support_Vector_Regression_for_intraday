import numpy as np
from processes_generators.arx_processes import *

def generate_sample_and_predictors(model, model_params_set, sample_len, seed):
    if model == 'AR':
        Y = generate_AR(model_params_set['alpha'], model_params_set['beta'], sample_len, seed)
        X_0 = np.ones(sample_len)
        Y = (Y - np.mean(Y))/np.std(Y)
        X = np.vstack((Y[:-1], X_0[1:])).T

    if model == 'AR_exp':
        Y = generate_AR(model_params_set['alpha'], model_params_set['beta'], sample_len, seed)
        X_0 = np.ones(sample_len)
        Y = (Y - np.mean(Y))/np.std(Y)
        Y = np.exp(Y) # exp is after the standardisation as we do not want any explosions
        Y = (Y - np.mean(Y))/np.std(Y)
        X = np.vstack((Y[:-1], X_0[1:])).T

    if model == 'ARX':
        rng = np.random.default_rng(seed=seed)
        X_1 = np.sin(np.arange(sample_len)/(sample_len/20)) + rng.normal(0, 0.1, sample_len)
        Y = generate_ARX(model_params_set['alpha'], model_params_set['beta'], [X_1], sample_len, seed)
        X_0 = np.ones(sample_len)
        Y = (Y - np.mean(Y))/np.std(Y)
        X_1 = (X_1 - np.mean(X_1))/np.std(X_1)
        X = np.vstack((Y[:-1], X_0[1:], X_1[1:])).T

    if model == 'ARX_exp':
        rng = np.random.default_rng(seed=seed)
        X_1 = np.sin(np.arange(sample_len)/(sample_len/20)) + rng.normal(0, 0.1, sample_len)
        Y = generate_ARX(model_params_set['alpha'], model_params_set['beta'], [X_1], sample_len, seed)
        X_0 = np.ones(sample_len)
        Y = (Y - np.mean(Y))/np.std(Y)
        X_1 = (X_1 - np.mean(X_1))/np.std(X_1)
        Y = np.exp(Y)
        Y = (Y - np.mean(Y))/np.std(Y)
        X = np.vstack((Y[:-1], X_0[1:], X_1[1:])).T

    if model == 'AR_noshock':
        Y = generate_AR(model_params_set['alpha'], model_params_set['beta'], sample_len, seed, shock=False)
        X_0 = np.ones(sample_len)
        Y = (Y - np.mean(Y))/np.std(Y)
        X = np.vstack((Y[:-1], X_0[1:])).T

    if model == 'AR_exp_noshock':
        Y = generate_AR(model_params_set['alpha'], model_params_set['beta'], sample_len, seed, shock=False)
        X_0 = np.ones(sample_len)
        Y = (Y - np.mean(Y))/np.std(Y)
        Y = np.exp(Y) # exp is after the standardisation as we do not want any explosions
        Y = (Y - np.mean(Y))/np.std(Y)
        X = np.vstack((Y[:-1], X_0[1:])).T

    if model == 'ARX_noshock':
        rng = np.random.default_rng(seed=seed)
        X_1 = np.sin(np.arange(sample_len)/(sample_len/20)) + rng.normal(0, 0.1, sample_len)
        Y = generate_ARX(model_params_set['alpha'], model_params_set['beta'], [X_1], sample_len, seed, shock=False)
        X_0 = np.ones(sample_len)
        Y = (Y - np.mean(Y))/np.std(Y)
        X_1 = (X_1 - np.mean(X_1))/np.std(X_1)
        X = np.vstack((Y[:-1], X_0[1:], X_1[1:])).T

    if model == 'ARX_exp_noshock':
        rng = np.random.default_rng(seed=seed)
        X_1 = np.sin(np.arange(sample_len)/(sample_len/20)) + rng.normal(0, 0.1, sample_len)
        Y = generate_ARX(model_params_set['alpha'], model_params_set['beta'], [X_1], sample_len, seed, shock=False)
        X_0 = np.ones(sample_len)
        Y = (Y - np.mean(Y))/np.std(Y)
        X_1 = (X_1 - np.mean(X_1))/np.std(X_1)
        Y = np.exp(Y)
        Y = (Y - np.mean(Y))/np.std(Y)
        X = np.vstack((Y[:-1], X_0[1:], X_1[1:])).T

    return X, Y
