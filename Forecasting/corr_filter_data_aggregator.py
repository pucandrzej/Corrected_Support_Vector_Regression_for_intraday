# aggregating the results of correlation based variables rejection in parallel (we have a lot of results files so reading them takes time, thus parallel solution)
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import pickle
import matplotlib as mpl
from multiprocessing import Pool
from scipy import stats

def load_delivery_results(inp):
    delivery, horizons, trade_times, results_dir = inp
    corr_rejections = {}
    for horizon in horizons:
        corr_rejections[horizon] = {}
        for trade_time in trade_times:
            fore_dir = f"2020-01-01_2020-12-31_427_{delivery}_[{horizon}]_{trade_time}_True"
            corr_rejections[horizon][trade_time] = {}

            if os.path.exists(os.path.join(results_dir, fore_dir)):
                rejections = [f for f in os.listdir(os.path.join(results_dir, fore_dir)) if '.csv' in f and 'test_' in f and '_11_weights' in f]

                if len(rejections):
                    df_sample = pd.read_csv(os.path.join(results_dir, fore_dir, rejections[0]))

                    rejected_perc_S1 = []
                    rejected_perc_S2 = []

                    removed_exog_idxs_S1 = []
                    removed_exog_idxs_S2 = []

                    exog_corr_pairs_S1 = []
                    exog_corr_pairs_S2 = []

                    counter_of_rejections_per_simu_case = {}

                    for rejected in rejections:
                        total_variables_no = int(rejected.split('_shape_of_underlying_')[1].split('_init_shape_')[0].split(', ')[1][:-1])
                        df = pd.read_csv(os.path.join(results_dir, fore_dir, rejected))
                        perc = len(np.unique(df['removed indices']))/total_variables_no # perc. of rejected
                        if perc > 1:
                            raise
                        for removed_idx in np.unique(df[df['removed indices'] > total_variables_no - 20]['removed indices']):
                            if '_S1_' in rejected:
                                removed_exog_idxs_S1.append(removed_idx - (total_variables_no - 20))
                                if removed_idx - (total_variables_no - 20) not in counter_of_rejections_per_simu_case.keys():
                                    counter_of_rejections_per_simu_case[removed_idx - (total_variables_no - 20)] = 1
                                else:
                                    counter_of_rejections_per_simu_case[removed_idx - (total_variables_no - 20)] += 1
                                    if counter_of_rejections_per_simu_case[removed_idx - (total_variables_no - 20)] > 366:
                                        raise ValueError("No. of exog rejections exceeds 366. U sure it works?")
                                exog_corr_pairs_S1.append((removed_idx - (total_variables_no - 20), df[df['removed indices'] == removed_idx]['correlated with'].to_numpy()[0] - (total_variables_no - 20) if df[df['removed indices'] == removed_idx]['correlated with'].to_numpy()[0] - (total_variables_no - 20) > 0 else df[df['removed indices'] == removed_idx]['correlated with'].to_numpy()[0]))
                            elif '_S2_' in rejected:
                                removed_exog_idxs_S2.append(removed_idx - (total_variables_no - 20))
                                exog_corr_pairs_S2.append((removed_idx - (total_variables_no - 20), df[df['removed indices'] == removed_idx]['correlated with'].to_numpy()[0] - (total_variables_no - 20) if df[df['removed indices'] == removed_idx]['correlated with'].to_numpy()[0] - (total_variables_no - 20) > 0 else df[df['removed indices'] == removed_idx]['correlated with'].to_numpy()[0]))

                        if '_S1_' in rejected:
                            rejected_perc_S1.append(perc)
                        elif '_S2_' in rejected:
                            rejected_perc_S2.append(perc)
                    corr_rejections[horizon][trade_time]['S1_perc'] = np.mean(rejected_perc_S1)
                    corr_rejections[horizon][trade_time]['S2_perc'] = np.mean(rejected_perc_S2)
                    corr_rejections[horizon][trade_time]['S1_exog_rejected'] = stats.mode(removed_exog_idxs_S1)[0]
                    corr_rejections[horizon][trade_time]['S1_exog_rejected_count'] = counter_of_rejections_per_simu_case
                    corr_rejections[horizon][trade_time]['S1_exog_corr_pair'] = np.unique(exog_corr_pairs_S1, axis=0)
                    corr_rejections[horizon][trade_time]['S2_exog_corr_pair'] = np.unique(exog_corr_pairs_S2, axis=0)

            if corr_rejections[horizon][trade_time] == {}:
                del corr_rejections[horizon][trade_time]

        if corr_rejections[horizon] == {}:
            del corr_rejections[horizon]
    if corr_rejections != {}:
        pickle.dump(corr_rejections, open(f'CORR_ANALYSIS/mae_results_{delivery}.pickle', 'wb'))

if __name__ == '__main__':
    results_dir = 'C:/Users/riczi/Studies/Continuous_market_analysis/Forecasting/CLEAN_RESULTS_PAPER/CORR_ANALYSIS/CORR_REJECTION_ANALYSIS/'
    results_dirs = os.listdir(results_dir)

    deliveries = np.unique([int(dir_name.split('_427_')[1].split('_')[0]) for dir_name in results_dirs])
    horizons = np.unique([int(dir_name.split('_427_')[1].split('_')[1].replace('[','').replace(']','')) for dir_name in results_dirs])

    trade_times = []
    for delivery in deliveries:
        delivery_dirs = [resul_dir for resul_dir in results_dirs if f'_{delivery}_' in resul_dir]
        for delivery_dir in delivery_dirs:
            trade_times.append(int(delivery_dir.split(f"]_")[1].split('_')[0]))
    trade_times = np.unique(trade_times)

    with Pool(processes=48) as p:
        inputlist = [(i, horizons, trade_times, results_dir) for i in deliveries]
        _ = p.map(load_delivery_results, inputlist)
