# calculates the MAE of kernel models in parallel
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import pickle
import matplotlib as mpl
from multiprocessing import Pool
from scipy import stats


def my_rmse(X, Y):
    return np.sqrt(((X - Y) ** 2).mean())


def my_mae(X, Y):
    return np.mean(np.abs(X - Y))


def dm_pval(actual, naive, col_results):
    d = np.abs(np.array(naive) - np.array(actual)) - np.abs(
        np.array(col_results) - np.array(actual)
    )
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=0)
    DM_stat = mean_d / np.sqrt((1 / len(d)) * var_d)
    return 1 - stats.norm.cdf(DM_stat)


def my_mape(X, Y):
    """X is the actual values vector"""
    # return 100*np.mean(np.abs(X - Y)/X)
    return np.nan


def load_delivery_results(inp):
    delivery, horizons, results_dir = inp
    for trade_vs_delivery_delta in [60]:
        mae_results = {}
        mape_results = {}
        for horizon in horizons:
            mae_results[horizon] = {}
            mape_results[horizon] = {}
            trade_time = delivery * 15 + 8 * 60 - trade_vs_delivery_delta
            mae_results[horizon][trade_time] = {}
            mape_results[horizon][trade_time] = {}

            fore_dir = f"kernel_hr_naive_mult_2020-01-01_2020-12-31_427_{delivery}_[{horizon}]_{trade_time}_True"
            random_forest_dir = f"RESULTS/DEVEL_RESULTS_RF/random_forest_2020-01-01_2020-12-31_427_{delivery}_[{horizon}]_{trade_time}_True"
            lasso_dir = f"LASSO_RECALC/lasso_2020-01-01_2020-12-31_427_{delivery}_[{horizon}]_{trade_time}_True"

            if os.path.exists(os.path.join(results_dir, fore_dir)):
                forecasts = [
                    f
                    for f in os.listdir(os.path.join(results_dir, fore_dir))
                    if ".csv" in f and "test_" in f
                ]

                if len(forecasts):
                    try:
                        df_sample = pd.read_csv(
                            os.path.join(results_dir, fore_dir, forecasts[0]),
                            encoding="unicode_escape",
                        )
                    except:
                        print(os.path.join(results_dir, fore_dir, forecasts[0]))

                    actual = []
                    naive = []
                    for fore in forecasts:
                        try:
                            df = pd.read_csv(
                                os.path.join(results_dir, fore_dir, fore),
                                encoding="unicode_escape",
                            )
                        except:
                            print(os.path.join(results_dir, fore_dir, fore))
                        try:
                            actual.append(df.loc[0, "actual"])
                            naive.append(df.loc[0, "naive"])
                        except:
                            # remove the empty file
                            os.remove(os.path.join(results_dir, fore_dir, fore))

                    if (
                        trade_time == delivery * 15 + 8 * 60 - 60
                    ):  # only for this time we have LASSO and RF
                        for col in [
                            "prediction",
                            "prediction_close",
                            "prediction_exog",
                            "prediction_32",
                            "prediction_33",
                            "prediction_34",
                            "prediction_35",
                        ]:
                            # uncomment for RF
                            col_results = []
                            for fore in forecasts:
                                try:
                                    df_random_forest = pd.read_csv(
                                        os.path.join(random_forest_dir, fore),
                                        encoding="unicode_escape",
                                    )
                                except:
                                    print(os.path.join(random_forest_dir, fore))
                                col_results.append(df_random_forest.loc[0, col])
                            mae_results[horizon][trade_time][col + "random_forest"] = (
                                my_mae(np.array(actual), np.array(col_results))
                            )
                            mape_results[horizon][trade_time][col + "random_forest"] = (
                                my_mape(np.array(actual), np.array(col_results))
                            )
                            # add avg with naive
                            mae_results[horizon][trade_time][
                                col + "_avg_with_naive" + "random_forest"
                            ] = my_mae(
                                np.array(actual),
                                (np.array(col_results) + np.array(naive)) / 2,
                            )
                            mape_results[horizon][trade_time][
                                col + "_avg_with_naive" + "random_forest"
                            ] = my_mape(
                                np.array(actual),
                                (np.array(col_results) + np.array(naive)) / 2,
                            )
                            # add also the info about DM test p-value
                            p_value = dm_pval(
                                actual,
                                naive,
                                (np.array(col_results) + np.array(naive)) / 2,
                            )
                            mae_results[horizon][trade_time][
                                col
                                + "_avg_with_naive_DM_wrt_naive_pval"
                                + "random_forest"
                            ] = p_value
                            p_value = dm_pval(actual, naive, col_results)
                            mae_results[horizon][trade_time][
                                col + "_DM_wrt_naive_pval" + "random_forest"
                            ] = p_value

                            col_results = []
                            for fore in forecasts:
                                try:
                                    df_lasso = pd.read_csv(
                                        os.path.join(lasso_dir, fore),
                                        encoding="unicode_escape",
                                    )
                                except:
                                    print(os.path.join(lasso_dir, fore))
                                col_results.append(df_lasso.loc[0, col])
                            mae_results[horizon][trade_time][col + "lasso"] = my_mae(
                                np.array(actual), np.array(col_results)
                            )
                            mape_results[horizon][trade_time][col + "lasso"] = my_mape(
                                np.array(actual), np.array(col_results)
                            )
                            # add avg with naive
                            mae_results[horizon][trade_time][
                                col + "_avg_with_naive" + "lasso"
                            ] = my_mae(
                                np.array(actual),
                                (np.array(col_results) + np.array(naive)) / 2,
                            )
                            mape_results[horizon][trade_time][
                                col + "_avg_with_naive" + "lasso"
                            ] = my_mape(
                                np.array(actual),
                                (np.array(col_results) + np.array(naive)) / 2,
                            )
                            # add also the info about DM test p-value
                            p_value = dm_pval(
                                actual,
                                naive,
                                (np.array(col_results) + np.array(naive)) / 2,
                            )
                            mae_results[horizon][trade_time][
                                col + "_avg_with_naive_DM_wrt_naive_pval" + "lasso"
                            ] = p_value
                            p_value = dm_pval(actual, naive, col_results)
                            mae_results[horizon][trade_time][
                                col + "_DM_wrt_naive_pval" + "lasso"
                            ] = p_value

                    if len(actual) == 366:
                        fore_cols = {}
                        for col in df_sample.columns:
                            try:
                                if "prediction" in col or col == "naive":
                                    col_results = []
                                    for fore in forecasts:
                                        try:
                                            df = pd.read_csv(
                                                os.path.join(
                                                    results_dir, fore_dir, fore
                                                ),
                                                encoding="unicode_escape",
                                            )
                                        except:
                                            print(
                                                os.path.join(
                                                    results_dir, fore_dir, fore
                                                )
                                            )
                                        col_results.append(df.loc[0, col])

                                    mae_results[horizon][trade_time][col] = my_mae(
                                        np.array(actual), np.array(col_results)
                                    )
                                    mape_results[horizon][trade_time][col] = my_mape(
                                        np.array(actual), np.array(col_results)
                                    )
                                    # add also the info about DM test p-value
                                    if col != "naive":
                                        # add avg with naive
                                        mae_results[horizon][trade_time][
                                            col + "_avg_with_naive"
                                        ] = my_mae(
                                            np.array(actual),
                                            (np.array(col_results) + np.array(naive))
                                            / 2,
                                        )
                                        mape_results[horizon][trade_time][
                                            col + "_avg_with_naive"
                                        ] = my_mape(
                                            np.array(actual),
                                            (np.array(col_results) + np.array(naive))
                                            / 2,
                                        )
                                        # add also the info about DM test p-value
                                        p_value = dm_pval(
                                            actual,
                                            naive,
                                            (np.array(col_results) + np.array(naive))
                                            / 2,
                                        )
                                        mae_results[horizon][trade_time][
                                            col + "_avg_with_naive_DM_wrt_naive_pval"
                                        ] = p_value
                                        p_value = dm_pval(actual, naive, col_results)
                                        mae_results[horizon][trade_time][
                                            col + "_DM_wrt_naive_pval"
                                        ] = p_value
                                        fore_cols[col] = col_results
                            except:
                                print(f"Skipped column {col}")

                        # mae_results[horizon][trade_time]['avg_0_1_7'] = my_mae(np.array(actual), (np.array(fore_cols['prediction_0']) + np.array(fore_cols['prediction_1']) + np.array(fore_cols['prediction_7']) + np.array(naive))/4)
                        # # add also the info about DM test p-value
                        # p_value = dm_pval(actual, naive, (np.array(fore_cols['prediction_0']) + np.array(fore_cols['prediction_1']) + np.array(fore_cols['prediction_7']) + np.array(naive))/4)
                        # mae_results[horizon][trade_time]['avg_0_1_7_DM_wrt_naive_pval'] = p_value

                        # mae_results[horizon][trade_time]['avg_0_1_7_no_naive'] = my_mae(np.array(actual), (np.array(fore_cols['prediction_0']) + np.array(fore_cols['prediction_1']) + np.array(fore_cols['prediction_7']))/3)
                        # # add also the info about DM test p-value
                        # p_value = dm_pval(actual, naive, (np.array(fore_cols['prediction_0']) + np.array(fore_cols['prediction_1']) + np.array(fore_cols['prediction_7']))/3)
                        # mae_results[horizon][trade_time]['avg_0_1_7_no_naive_DM_wrt_naive_pval'] = p_value

                        mae_results[horizon][trade_time]["avg_1_2_7_naive"] = my_mae(
                            np.array(actual),
                            (
                                np.array(fore_cols["prediction_1"])
                                + np.array(fore_cols["prediction_2"])
                                + np.array(fore_cols["prediction_7"])
                                + np.array(naive)
                            )
                            / 4,
                        )
                        mape_results[horizon][trade_time]["avg_1_2_7_naive"] = my_mape(
                            np.array(actual),
                            (
                                np.array(fore_cols["prediction_1"])
                                + np.array(fore_cols["prediction_2"])
                                + np.array(fore_cols["prediction_7"])
                                + np.array(naive)
                            )
                            / 4,
                        )

                        # add also the info about DM test p-value
                        p_value = dm_pval(
                            actual,
                            naive,
                            (
                                np.array(fore_cols["prediction_1"])
                                + np.array(fore_cols["prediction_2"])
                                + np.array(fore_cols["prediction_7"])
                                + np.array(naive)
                            )
                            / 4,
                        )
                        mae_results[horizon][trade_time][
                            "avg_1_2_7_naive_DM_wrt_naive_pval"
                        ] = p_value

                        # mae_results[horizon][trade_time]['avg_0_1_6_7_no_naive'] = my_mae(np.array(actual), (np.array(fore_cols['prediction_0']) + np.array(fore_cols['prediction_1']) + np.array(fore_cols['prediction_6']) + np.array(fore_cols['prediction_7']))/4)
                        # # add also the info about DM test p-value
                        # p_value = dm_pval(actual, naive, (np.array(fore_cols['prediction_0']) + np.array(fore_cols['prediction_1']) + np.array(fore_cols['prediction_6']) + np.array(fore_cols['prediction_7']))/4)
                        # mae_results[horizon][trade_time]['avg_0_1_6_7_no_naive_DM_wrt_naive_pval'] = p_value

                        # mae_results[horizon][trade_time]['avg_1_2'] = my_mae(np.array(actual), (np.array(fore_cols['prediction_1']) + np.array(fore_cols['prediction_2']) + np.array(naive))/3)
                        # # add also the info about DM test p-value
                        # p_value = dm_pval(actual, naive, (np.array(fore_cols['prediction_1']) + np.array(fore_cols['prediction_2']) + np.array(naive))/3)
                        # mae_results[horizon][trade_time]['avg_1_2_DM_wrt_naive_pval'] = p_value

            if mae_results[horizon][trade_time] == {}:
                del mae_results[horizon][trade_time]
                del mape_results[horizon][trade_time]

            if mae_results[horizon] == {}:
                del mae_results[horizon]
                del mape_results[horizon]

        if mape_results != {}:
            pickle.dump(
                mae_results,
                open(
                    f"MAE_ANALYSIS_FINAL/mae_results_{trade_vs_delivery_delta}_{delivery}.pickle",
                    "wb",
                ),
            )
            pickle.dump(
                mape_results,
                open(
                    f"MAE_ANALYSIS_FINAL/mape_results_{trade_vs_delivery_delta}_{delivery}.pickle",
                    "wb",
                ),
            )


if __name__ == "__main__":
    results_dir = "RESULTS/DEVEL_RESULTS_l"
    results_dirs = os.listdir(results_dir)

    deliveries = np.arange(96)
    horizons = [30, 60, 90, 120, 150, 180, 210, 300, 390, 480]

    with Pool(processes=32) as p:
        inputlist = [(i, horizons, results_dir) for i in deliveries]
        _ = p.map(load_delivery_results, inputlist)
