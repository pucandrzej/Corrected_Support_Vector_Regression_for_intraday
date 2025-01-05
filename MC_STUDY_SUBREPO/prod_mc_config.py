import numpy as np

prod_mc_config = {

    'mc_steps': 90,
    'mc_steps_features_sens': 90,
    'sample_lengths': [101, 201],
    'models': ['ARX_noshock', 'ARX_exp_noshock', 'ARX', 'ARX_exp', 'AR_noshock', 'AR_exp_noshock', 'AR', 'AR_exp'],
    'model_params_sets': {

        'AR': [{'alpha': 0.1, 'beta': 0.1}, {'alpha': 0.5, 'beta': 0.1}, {'alpha': 0.9, 'beta': 0.1}],
        'AR_exp': [{'alpha': 0.1, 'beta': 0.1}, {'alpha': 0.5, 'beta': 0.1}, {'alpha': 0.9, 'beta': 0.1}],
        'ARX': [{'alpha': 0.1, 'beta': 0.1}, {'alpha': 0.5, 'beta': 0.1}, {'alpha': 0.9, 'beta': 0.1}],
        'ARX_exp': [{'alpha': 0.1, 'beta': 0.1}, {'alpha': 0.5, 'beta': 0.1}, {'alpha': 0.9, 'beta': 0.1}],
        'AR_noshock': [{'alpha': 0.1, 'beta': 0.1}, {'alpha': 0.5, 'beta': 0.1}, {'alpha': 0.9, 'beta': 0.1}],
        'AR_exp_noshock': [{'alpha': 0.1, 'beta': 0.1}, {'alpha': 0.5, 'beta': 0.1}, {'alpha': 0.9, 'beta': 0.1}],
        'ARX_noshock': [{'alpha': 0.1, 'beta': 0.1}, {'alpha': 0.5, 'beta': 0.1}, {'alpha': 0.9, 'beta': 0.1}],
        'ARX_exp_noshock': [{'alpha': 0.1, 'beta': 0.1}, {'alpha': 0.5, 'beta': 0.1}, {'alpha': 0.9, 'beta': 0.1}],
    },
    'param_grid_slide_on_median': {
        'q_kernel': np.linspace(0.51, 0.99, 10), # symmetric, so checking < 0.5 makes no sense
        'q_kernel_naive': np.linspace(0.51, 0.99, 10),
        'q_kernel_div': np.linspace(0.51, 0.99, 10),
        'div_impact': np.linspace(0.1, 1, 4),
        'epsilon': np.linspace(0, 0.5, 8),
        'C': [0.1, 1, 3],
        'q_data_div': [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
        'q_data': [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
        'q_data_naive': [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
    }
}
