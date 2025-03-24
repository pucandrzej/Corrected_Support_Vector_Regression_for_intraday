# main MC simulation script
# consists of
# - loading the simulation parameters
# - running the simulation loop
# - invoking the results saving scripts

from processes_generators.generate_sample_and_predictors import (
    generate_sample_and_predictors,
)
from tools.generate_mc_steps_seeds import generate_mc_steps_seeds
from tools.plot_cv_results import plot_cv_results
from processes_predictors.predict import predict
from prod_mc_config import prod_mc_config
from processes_predictors.prepare_params_cv import prepare_params_combinations
from processes_predictors.parallel_cv import parallel_cv
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import argparse
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pickle
from wait_for_low_cpu_load import wait_for_low_cpu

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    default="dev",
    help="Specifies whether to use the devel or prod confid files.",
)
parser.add_argument(
    "--cv_run",
    default=False,
    action="store_true",
    help="To use CV values if True or hand picked when False.",
)
parser.add_argument(
    "--cv_recalc",
    default=False,
    action="store_true",
    help="To recalculate CV values if True or oad from CV cache when False.",
)
parser.add_argument(
    "--use_base_forecast_corr",
    default=False,
    action="store_true",
    help="Use the base model forecasts for kernel correction.",
)
args = parser.parse_args()


def mc_step(params):
    model, model_params, sample_len, step_seed, best_params, idx = params

    X, Y = generate_sample_and_predictors(model, model_params, sample_len, step_seed)

    step_results_df = pd.DataFrame()

    base_model_fores = []
    svr_fores = []
    csvr_fores = []
    ccsvr_fores = []

    svr_alignments = []
    csvr_alignments = []
    ccsvr_alignments = []

    base_model_ims = []
    svr_ims = []
    csvr_ims = []
    ccsvr_ims = []

    Y_fore = []
    for i in range(sample_len // 2, sample_len - 1):
        X_train = X[:i, :]
        Y_train = Y[1 : i + 1]
        X_test = X[np.newaxis, i, :]
        Y_fore.append(Y[i + 1])

        # base model estimation
        base_model_forecast, base_model_insample_mae, _, base_model_forecasts = predict(
            model, X_train.copy(), Y_train.copy(), X_test.copy()
        )

        if args.use_base_forecast_corr:
            kernel_correction_fore = base_model_forecast
            kernel_correction_fores = base_model_forecasts
        else:
            kernel_correction_fore = Y[i]
            kernel_correction_fores = Y[:i]

        # SVR model estimation
        svr_forecast, svr_insample_mae, svr_kernel_alignment, _ = predict(
            "SVR",
            X_train.copy(),
            Y_train.copy(),
            X_test.copy(),
            q_kernel=best_params["SVR"][idx]["q_kernel"],
            q_data=best_params["SVR"][idx]["q_data"],
            epsilon=best_params["SVR"][idx]["epsilon"],
            C=best_params["SVR"][idx]["C"],
        )

        # cSVR
        csvr_forecast, csvr_insample_mae, csvr_kernel_alignment, _ = predict(
            "cSVR",
            X_train.copy(),
            Y_train.copy(),
            X_test.copy(),
            q_kernel=best_params["cSVR"][idx]["q_kernel"],
            q_data=best_params["cSVR"][idx]["q_data"],
            naive_forecasts=kernel_correction_fores.copy(),
            naive_forecast=kernel_correction_fore.copy(),
            q_kernel_naive=best_params["cSVR"][idx]["q_kernel_naive"],
            q_data_naive=best_params["cSVR"][idx]["q_data_naive"],
            epsilon=best_params["cSVR"][idx]["epsilon"],
            C=best_params["cSVR"][idx]["C"],
        )

        # ccSVR
        ccsvr_forecast, ccsvr_insample_mae, ccsvr_kernel_alignment, _ = predict(
            "ccSVR",
            X_train.copy(),
            Y_train.copy(),
            X_test.copy(),
            q_kernel=best_params["ccSVR"][idx]["q_kernel"],
            q_data=best_params["ccSVR"][idx]["q_data"],
            naive_forecasts=kernel_correction_fores.copy(),
            naive_forecast=kernel_correction_fore.copy(),
            q_kernel_naive=best_params["ccSVR"][idx]["q_kernel_naive"],
            q_data_naive=best_params["ccSVR"][idx]["q_data_naive"],
            q_kernel_div=best_params["ccSVR"][idx]["q_kernel_div"],
            q_data_div=best_params["ccSVR"][idx]["q_data_div"],
            div_impact=best_params["ccSVR"][idx]["div_impact"],
            epsilon=best_params["ccSVR"][idx]["epsilon"],
            C=best_params["ccSVR"][idx]["C"],
        )

        # collect all parameters from forecast iteration
        base_model_fores.append(base_model_forecast)
        svr_fores.append(svr_forecast)
        csvr_fores.append(csvr_forecast)
        ccsvr_fores.append(ccsvr_forecast)

        base_model_ims.append(base_model_insample_mae)
        svr_ims.append(svr_insample_mae)
        csvr_ims.append(csvr_insample_mae)
        ccsvr_ims.append(ccsvr_insample_mae)

        svr_alignments.append(svr_kernel_alignment)
        csvr_alignments.append(csvr_kernel_alignment)
        ccsvr_alignments.append(ccsvr_kernel_alignment)

    # calculate the MAE for each model
    step_results_df[f"{model}_MAE"] = [
        np.mean(np.abs(np.array(Y_fore) - np.array(base_model_fores)))
    ]
    step_results_df[f"SVR_MAE"] = [
        np.mean(np.abs(np.array(Y_fore) - np.array(svr_fores)))
    ]
    step_results_df[f"cSVR_MAE"] = [
        np.mean(np.abs(np.array(Y_fore) - np.array(csvr_fores)))
    ]
    step_results_df[f"ccSVR_MAE"] = [
        np.mean(np.abs(np.array(Y_fore) - np.array(ccsvr_fores)))
    ]

    # average insample MAE
    step_results_df[f"{model}_insample_MAE"] = [np.mean(base_model_ims)]
    step_results_df[f"SVR_insample_MAE"] = [np.mean(svr_ims)]
    step_results_df[f"cSVR_insample_MAE"] = [np.mean(csvr_ims)]
    step_results_df[f"ccSVR_insample_MAE"] = [np.mean(ccsvr_ims)]

    # average kernel alignments
    step_results_df["SVR_kernel_alignment"] = np.mean(svr_alignments)
    step_results_df["cSVR_kernel_alignment"] = np.mean(csvr_alignments)
    step_results_df["ccSVR_kernel_alignment"] = np.mean(ccsvr_alignments)

    return step_results_df


if __name__ == "__main__":
    print("Waiting till the cpu load is small...")
    wait_for_low_cpu()
    print("Running the simulation.")

    start = time.time()

    mc_config = prod_mc_config

    mc_steps = mc_config["mc_steps"]
    seeds = generate_mc_steps_seeds(mc_steps)

    sample_lengths = mc_config["sample_lengths"]

    for model in mc_config["models"]:
        for sample_length in sample_lengths:
            for params_index, models_params in enumerate(
                mc_config["model_params_sets"][model]
            ):
                dir_name = (
                    f"{model}_params_set_{params_index}_sample_length_{sample_length}"
                )
                os.makedirs(os.path.join("MC_RESULTS", dir_name), exist_ok=True)

                # prepare the CV based parameters
                all_mae_cv = {}
                all_mae_cv["SVR"] = []
                all_mae_cv["cSVR"] = []
                all_mae_cv["ccSVR"] = []
                best_params = {}
                best_params["SVR"] = []
                best_params["cSVR"] = []
                best_params["ccSVR"] = []
                for mc_step_cv in range(mc_steps):
                    X, Y = generate_sample_and_predictors(
                        model, models_params, sample_length, seeds[mc_step_cv]
                    )
                    i = sample_len // 2 - 1
                    X_train = X[:i, :]
                    Y_train = Y[1 : i + 1]
                    # TimeSeriesSplit for Cross-Validation
                    tscv = TimeSeriesSplit(n_splits=20, test_size=1)

                    kernel_model = "SVR"
                    SVR_parameters = prepare_params_combinations(
                        kernel_model, mc_config["param_grid_slide_on_median"]
                    )
                    params_cv = [
                        (
                            kernel_model,
                            model,
                            X_train,
                            Y_train,
                            tscv,
                            args.use_base_forecast_corr,
                            Y[:i],
                            param,
                        )
                        for param in SVR_parameters
                    ]
                    with Pool(processes=min(30, len(SVR_parameters))) as p:
                        results_svr = p.map(parallel_cv, params_cv.copy())
                    # get the best params
                    best_params[kernel_model].append(
                        SVR_parameters[np.argmin(results_svr)]
                    )
                    all_mae_cv[kernel_model].append(results_svr)

                    # use the SVR best models to also search for the cSVR best models
                    kernel_model = "cSVR"
                    cSVR_parameters = prepare_params_combinations(
                        kernel_model,
                        mc_config["param_grid_slide_on_median"],
                        best_params_svr=best_params["SVR"][-1],
                    )
                    params_cv = [
                        (
                            kernel_model,
                            model,
                            X_train,
                            Y_train,
                            tscv,
                            args.use_base_forecast_corr,
                            Y[:i],
                            param,
                        )
                        for param in cSVR_parameters
                    ]
                    with Pool(processes=min(30, len(cSVR_parameters))) as p:
                        results_csvr = p.map(parallel_cv, params_cv.copy())
                    # get the best params
                    best_params[kernel_model].append(
                        cSVR_parameters[np.argmin(results_csvr)]
                    )
                    all_mae_cv[kernel_model].append(results_csvr)

                    # use the cSVR best models to also search for the ccSVR best models
                    kernel_model = "ccSVR"
                    ccSVR_parameters = prepare_params_combinations(
                        kernel_model,
                        mc_config["param_grid_slide_on_median"],
                        best_params_svr=best_params["cSVR"][-1],
                    )
                    params_cv = [
                        (
                            kernel_model,
                            model,
                            X_train,
                            Y_train,
                            tscv,
                            args.use_base_forecast_corr,
                            Y[:i],
                            param,
                        )
                        for param in ccSVR_parameters
                    ]
                    with Pool(processes=min(30, len(ccSVR_parameters))) as p:
                        results_ccsvr = p.map(parallel_cv, params_cv.copy())
                    # get the best params
                    best_params[kernel_model].append(
                        ccSVR_parameters[np.argmin(results_ccsvr)]
                    )
                    all_mae_cv[kernel_model].append(results_ccsvr)

                # plot the cv results and save them
                plot_cv_results(
                    {
                        "SVR": SVR_parameters.copy(),
                        "cSVR": cSVR_parameters.copy(),
                        "ccSVR": ccSVR_parameters.copy(),
                    },
                    all_mae_cv.copy(),
                    model,
                    sample_length,
                    params_index,
                )

                params = [
                    (model, models_params, sample_length, seeds[idx], best_params, idx)
                    for idx in range(mc_steps)
                ]
                with Pool(processes=min(30, mc_steps)) as p:
                    results = p.map(mc_step, params.copy())

                # save_results(results, model, models_params)
                current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                pickle.dump(
                    {
                        "SVR": SVR_parameters,
                        "cSVR": cSVR_parameters,
                        "ccSVR": ccSVR_parameters,
                    },
                    open(
                        os.path.join(
                            "MC_RESULTS",
                            dir_name,
                            current_datetime + "_CV_parameters.pickle",
                        ),
                        "wb",
                    ),
                )
                pickle.dump(
                    all_mae_cv,
                    open(
                        os.path.join(
                            "MC_RESULTS", dir_name, current_datetime + "_CV_mae.pickle"
                        ),
                        "wb",
                    ),
                )
                pd.DataFrame([pd.concat(results).mean(axis=0)]).to_csv(
                    os.path.join("MC_RESULTS", dir_name, current_datetime + ".csv")
                )

    end = time.time()
    print(f"Total simulation time: {end - start}")
