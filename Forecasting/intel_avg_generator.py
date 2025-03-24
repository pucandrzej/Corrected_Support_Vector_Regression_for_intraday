"""
postprocessing file to get the MAE weighted forecasts
scripts that generates the intel avg (avg based on the MAE on calibration window)
calculate the MAE of each model in the calibration window
"""

import os

try:
    os.chdir("Forecasting")  # for debuger run
except:
    pass

import pandas as pd
from datetime import timedelta
import numpy as np
from multiprocessing import Pool

cols_sets_to_average = [
    # ["prediction_0", "prediction_1", "prediction_7", "naive"],
    # ["prediction_0", "prediction_1", "prediction_7", "prediction_6"],
    # ["prediction_0", "prediction_1", "prediction_7", "prediction_6", "naive"],
    # ["prediction_1", "prediction_2", "prediction_7", "prediction_6", "naive"],
    # ["prediction_1", "prediction_2", "prediction_7", "prediction_6"],
    ["prediction_1", "prediction_2", "prediction_7", "naive"]
]
models = ["kernel_hr_naive_mult"]

calibration_window_lens = [7, 14, 21, 28]
dates = pd.date_range("2020-01-01", "2020-12-31")


def my_mae(X, Y):
    return np.mean(np.abs(X - Y))


def load_delivery_results(inp):
    delivery, horizons = inp
    print(f"Processing: {delivery}")
    for model in models:
        for trade_vs_delivery_delta in [180]:
            for forecasting_horizon in horizons:
                col_idx = 20
                for cols_to_average in cols_sets_to_average:
                    if (
                        model != "kernel_hr_naive_mult"
                    ):  # for LASSO and RF avg only the existing columns corresponding to
                        cols_to_average = [
                            "prediction",
                            "prediction_close",
                            "prediction_exog",
                            "naive",
                        ]
                    if model == "lasso":
                        results_folname = "C:/Users/riczi/Studies/Continuous_market_analysis/Forecasting/LASSO_results"
                    elif model == "random_forest":
                        results_folname = "C:/Users/riczi/Studies/Continuous_market_analysis/Forecasting/DEVEL_RESULTS_RF"
                    elif model == "kernel_hr_naive_mult":
                        results_folname = "C:/Users/riczi/Studies/Continuous_market_analysis/Forecasting/DEVEL_RESULTS_l"
                    for calibration_window_len in calibration_window_lens:
                        trade_time = delivery * 15 + 8 * 60 - trade_vs_delivery_delta
                        calib_weights_mul = {}
                        try:
                            for test_date in dates:
                                forecast_frames = []
                                for calib_date in pd.date_range(
                                    test_date - timedelta(calibration_window_len),
                                    test_date - timedelta(1),
                                ):
                                    if calib_date < pd.to_datetime(
                                        "2020-01-01"
                                    ):  # covered by a calibration run
                                        calibration_flag = "calibration"
                                    else:
                                        calibration_flag = "test"

                                    forecast_frames.append(
                                        pd.read_csv(
                                            f"{results_folname}/{model}_2020-01-01_2020-12-31_427_{delivery}_[{forecasting_horizon}]_{trade_time}_True/{calibration_flag}_{str((pd.to_datetime(calib_date) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=int(trade_time))).replace(':', ';')}_{forecasting_horizon}_11_weights_1.0_window_expanding.csv"
                                        )
                                    )

                                errors = []
                                for col_to_avg in cols_to_average:
                                    forecast = []
                                    actual = []
                                    for i_df, df in enumerate(forecast_frames):
                                        actual.append(df["actual"][0])
                                        forecast.append(df[col_to_avg][0])
                                    errors.append(
                                        my_mae(np.array(forecast), np.array(actual))
                                    )

                                # multiplicative weights
                                weights = []
                                for i in range(len(errors)):
                                    weights.append(
                                        (1 / errors[i]) / np.sum(1 / np.array(errors))
                                    )

                                calib_weights_mul[test_date] = weights

                            # create the average forecasts based on the weights
                            for _, date in enumerate(dates):
                                forecast = pd.read_csv(
                                    f"{results_folname}/{model}_2020-01-01_2020-12-31_427_{delivery}_[{forecasting_horizon}]_{trade_time}_True/test_{str((pd.to_datetime(date) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=int(trade_time))).replace(':', ';')}_{forecasting_horizon}_11_weights_1.0_window_expanding.csv",
                                    index_col=0,
                                )

                                avg_result = forecast.copy()

                                weights = calib_weights_mul[date]
                                avg_result[f"prediction_{12 + col_idx}"] = [
                                    np.sum(
                                        avg_result[cols_to_average].to_numpy()[0]
                                        * np.array(weights)
                                    )
                                ]

                                avg_result.to_csv(
                                    f"{results_folname}/{model}_2020-01-01_2020-12-31_427_{delivery}_[{forecasting_horizon}]_{trade_time}_True/test_{str((pd.to_datetime(date) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=int(trade_time))).replace(':', ';')}_{forecasting_horizon}_11_weights_1.0_window_expanding.csv"
                                )

                            col_idx += 1
                        except Exception as err:
                            print(f"Failed due to {err}")


if __name__ == "__main__":
    deliveries = np.arange(96)  # np.unique(
    #     [int(dir_name.split("_427_")[1].split("_")[0]) for dir_name in results_dirs]
    # )
    horizons = [30, 60, 90, 120, 150, 180, 210, 300, 390, 480]
    # horizons = np.unique(
    #     [
    #         int(
    #             dir_name.split("_427_")[1]
    #             .split("_")[1]
    #             .replace("[", "")
    #             .replace("]", "")
    #         )
    #         for dir_name in results_dirs
    #     ]
    # )

    inputlist = [(i, horizons) for i in deliveries]

    # load_delivery_results(inputlist[0])

    with Pool(processes=32) as p:
        _ = p.map(load_delivery_results, inputlist)
