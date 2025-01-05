import numpy as np
from processes_predictors.predict import predict

def prepare_params_combinations(model, param_grid, best_params_svr=None):
    params_combinations = []
    if model == 'SVR':
        for q_data in param_grid['q_data']:
            for q_kernel in param_grid['q_kernel']:
                for epsilon in param_grid['epsilon']:
                    for C in param_grid['C']:
                        params_combinations.append(
                            {
                                'q_kernel': q_kernel, 
                                'q_data': q_data, 
                                'epsilon': epsilon, 
                                'C': C
                            })

    if model == 'cSVR':
        q_data = best_params_svr['q_data']
        q_kernel = best_params_svr['q_kernel']
        epsilon = best_params_svr['epsilon']
        C = best_params_svr['C']
        for q_data_naive in param_grid['q_data_naive']:
            for q_kernel_naive in param_grid['q_kernel_naive']:
                params_combinations.append(
                    {
                        'q_kernel': q_kernel, 
                        'q_data': q_data, 
                        'q_kernel_naive': q_kernel_naive, 
                        'q_data_naive': q_data_naive, 
                        'epsilon': epsilon, 
                        'C': C
                    })

    if model == 'ccSVR':
        q_data = best_params_svr['q_data']
        q_kernel = best_params_svr['q_kernel']
        epsilon = best_params_svr['epsilon']
        C = best_params_svr['C']
        q_data_naive = best_params_svr['q_data_naive']
        q_kernel_naive = best_params_svr['q_kernel_naive']
        for q_data_div in param_grid['q_data_div']:
            for q_kernel_div in param_grid['q_kernel_div']:
                for div_impact in param_grid['div_impact']:
                    params_combinations.append(
                        {
                            'q_kernel': q_kernel, 
                            'q_data': q_data, 
                            'q_kernel_naive': q_kernel_naive, 
                            'q_data_naive': q_data_naive, 
                            'q_kernel_div': q_kernel_div, 
                            'q_data_div': q_data_div, 
                            'div_impact': div_impact, 
                            'epsilon': epsilon, 
                            'C': C
                        })

    return params_combinations
