'''
File contains the code for LASSO forecast of continuous market prices on DE intraday market.

Cool idea of relaxed lasso:
https://stats.stackexchange.com/questions/37989/advantages-of-doing-double-lasso-or-performing-lasso-twice
'''

import pandas as pd
import numpy as np
import os
import sys
try:
    os.chdir('Forecasting') # for debuger run
except:
    pass
from numba import njit
from datetime import datetime, timedelta
import argparse
from plot_correlation import plot_correlation
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from multiprocessing import Pool, RawArray
import scipy
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoLarsCV, LassoLars
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_validate
from sklearn.kernel_ridge import KernelRidge
from sklearn.compose import make_column_transformer
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.ensemble import BaggingRegressor
from preprocess_data import preprocess_data
import neural_tangents as nt
from neural_tangents import stax
import sqlite3
import pickle
import scipy.optimize as opt
import tensorflow as tf
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

import jax.numpy as jnp

from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad, vmap

from numba import jit as nb_jit
from remove_zerovar import remove_zerovar

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='kernel_hr_naive_mult', help='Model for forecasting')
parser.add_argument('--daterange_start', default='2020-01-01', help='Start of eval data')
parser.add_argument('--daterange_end', default='2020-12-31', help='End of eval data')
parser.add_argument('--lookback', default=365+62, help='Training window length')
parser.add_argument('--delivery_time', default=32, help='Index from 0 to 95 of the delivery quarter of the day')
parser.add_argument('--forecasting_horizons', default=[30,60,90,120,150,180,210,300,390,480], help='Minutes from the forecast to the trade that we want to forecast price for')
parser.add_argument('--trade_time', default=32*15 + 8*60 - 60, help='Minute before the delivery that we want to trade in')
parser.add_argument('--remove_zero_var', default=True, help='Remove the zero variance variables')
parser.add_argument('--weighting_alpha', default=1, help='Weighting coef')
parser.add_argument('--variable_set', default=11, help='Variable set choice')
parser.add_argument('--rolling_window', default=False, action='store_true', help='Use rolling window instead of the expanding window.')
parser.add_argument('--window_len', default=180, help='Training window length for rolling window.')
parser.add_argument('--calibration_window_len', default=28, help='For every date consider a historical results from a calibration window.')
parser.add_argument('--kernel_solver', default='SVR', help='Model to use: KRR or SVR')
parser.add_argument('--consider_exog', default=True, action='store_true', help='Consider the exog. variables')
parser.add_argument('--save_corr_rejected_variables', default=False, action='store_true', help='Save the indices of variables rejected in course of correlation filtering')
parser.add_argument('--no_filter_exog', default=True, action='store_true', help='Apply the correlation based filtering to exog. variables.')
parser.add_argument('--processes', default=1, help='No of processes')

svr_epsilon = 0.1
C = 1.0
preprocess_option = 0  # 0 - standardise, 1 - normalize to [0,1]
kernel_power = 2
alpha_KRR = 0.5
# dates for visual analysis
dates_for_analysis = [datetime(year = 2020, month = 1, day = 1), datetime(year = 2020, month = 2, day = 15), datetime(year = 2020, month = 5, day = 15), datetime(year = 2020, month = 8, day = 15), datetime(year = 2020, month = 11, day = 15)]

args = parser.parse_args()
kernel_model = args.kernel_solver
model = args.model
start = args.daterange_start
end = args.daterange_end
lookback = int(args.lookback)
weighting_alpha = float(args.weighting_alpha)
variable_set = int(args.variable_set)
forecasting_horizons = args.forecasting_horizons
trade_time = int(args.trade_time)
delivery_time = int(args.delivery_time)
remove_zero_var = args.remove_zero_var
window_len = int(args.window_len)
if not args.rolling_window: # if not rolling window the window is expanding
    window_len = 'expanding'
calibration_window_len = int(args.calibration_window_len)

results_folname = 'DEVEL_RESULTS_TIME_CALC'

if not os.path.exists(os.path.join(results_folname)):
    os.mkdir(os.path.join(results_folname))
for forecasting_horizon in forecasting_horizons:
    if not os.path.exists(os.path.join(results_folname, f'{model}_{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}')):
        os.mkdir(os.path.join(results_folname, f'{model}_{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}'))

corr_filter_results_folname = 'DEVEL_RESULTS_TIME_CALC/CORR_REJECTION_ANALYSIS'

if not os.path.exists(os.path.join(corr_filter_results_folname)):
    os.mkdir(os.path.join(corr_filter_results_folname))
for forecasting_horizon in forecasting_horizons:
    if not os.path.exists(os.path.join(corr_filter_results_folname, f'{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}')):
        os.mkdir(os.path.join(corr_filter_results_folname, f'{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}'))

dates = pd.date_range(start, end)
dates_calibration = pd.date_range(pd.to_datetime(start) - timedelta(days=calibration_window_len), pd.to_datetime(start) - timedelta(days=1))

# DA quarterhourly prices
DA_qtrly = pd.read_csv("../Data/Day-Ahead-Quarterly-Data/DA_prices_qtrly_2018_2020_preprocessed.csv", parse_dates = ['Datetime from'], na_values=['n/e'], index_col=['Datetime from'])
DA_qtrly.index = pd.to_datetime(DA_qtrly.index, dayfirst=True)
DA_qtrly = DA_qtrly['Day-ahead Price [EUR/MWh]'].to_frame(name='Day-ahead Price [EUR/MWh]')
DA_qtrly = DA_qtrly[DA_qtrly.index >= datetime(year=2018,month=10,day=31)]
DA_qtrly=DA_qtrly.interpolate()

# ID quarterhourly prices
ID_qtrly = pd.read_csv("../Data/ID_auction_preprocessed/ID_auction_price_2018-2020_preproc.csv", index_col=0, parse_dates=True)
ID_qtrly = ID_qtrly[ID_qtrly.index >= datetime(year=2018,month=10,day=31)]

# load Load
Load = pd.read_csv("../Data/Load/Load_2018-2020.csv", parse_dates = ['Time from'], index_col=0)
Load.index = pd.to_datetime(Load.index)
Load = Load[Load.index >= datetime(year=2018,month=10,day=31)]
Load=Load.interpolate()

# res generation
gen = pd.read_csv("../Data/Generation/Generation_2018-2020.csv", index_col=0)
gen.index = pd.to_datetime(gen.index)
gen = gen[gen.index >= datetime(year=2018,month=10,day=31)]
gen=gen.interpolate()

# cross border trade with FR
ge_fr = pd.read_csv("../Data/Crossborder/crossborder_ge_fr_2018-2020.csv", parse_dates = ['Time from'], index_col=0)
ge_fr.index = pd.to_datetime(ge_fr.index)
ge_fr = ge_fr[ge_fr.index >= datetime(year=2018,month=10,day=31)]
ge_fr=ge_fr.interpolate()

@nb_jit
def my_mae(X, Y):
	return np.mean(np.abs(X - Y))

# A global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(Model_data, Model_data_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['Model_data'] = Model_data
    var_dict['Model_data_shape'] = Model_data_shape

def run_one_day(inp):
    import keras
    idx = inp[0]
    date_fore = inp[1]
    forecasting_horizon = inp[2]
    variables_set = inp[3]
    model = inp[4]
    calibration_flag = inp[5]
    if not args.rolling_window:
        daily_data_window = np.swapaxes(np.frombuffer(var_dict['Model_data']).reshape(var_dict['Model_data_shape']), 1, 2)[:idx,:,:] # swapaxes needed after migration from np array to database
    else:
        daily_data_window = np.swapaxes(np.frombuffer(var_dict['Model_data']).reshape(var_dict['Model_data_shape']), 1, 2)[idx - window_len:idx,:,:] # swapaxes needed after migration from np array to database, -8 is for 7 week days and for python indexing
    # difference the data, assume 20min is the data avail. delay (source: market description)
    X = daily_data_window[:,:-1,20 + forecasting_horizon:-(20 + forecasting_horizon)] - daily_data_window[:,:-1,:-2*(20 + forecasting_horizon)] # load data without dummies
    dummies_col = daily_data_window[:,-1,-1] # load dummies to separate array

    Y = daily_data_window[:,delivery_time,-1] - daily_data_window[:,delivery_time,-(20 + forecasting_horizon)]

    # extract the volume for forecasted delivery
    volume_unaggr_undiff = daily_data_window[:,96+delivery_time,20 + forecasting_horizon:-(20 + forecasting_horizon)]
    # aggregate the traded volume by a sum
    X = np.hstack((X[:,:96,:],np.expand_dims(np.sum(X[:,96:,:],1),1)))

    # train the model & forecast
    results = pd.DataFrame()

    X_exog = X[:, -1:,:] # all exogenous variables (starting with agg volume)

    if variable_set == 15:
        X_close = np.hstack([
                                X[:,delivery_time - 4:delivery_time+1,-60:],
                                X[:,32:37 + 1,-60:]
                            ])
    else:
        X_close = X[:,np.max([delivery_time-4, 0]):np.min([delivery_time+1,96]),-60:]

    # You can define Your variables set here
    if variables_set == 10:
        X = X[:,delivery_time:delivery_time + 1,np.append(np.arange(0,np.shape(X)[-1]-1,15),np.shape(X)[-1]-1)]
    if variables_set == 11:
        X = X[:,np.max([delivery_time-8, 0]):np.min([delivery_time+5,96]),np.append(np.arange(0,np.shape(X)[-1]-1,15),np.shape(X)[-1]-1)]

    initial_shape_X = np.shape(X)
    initial_shape_X_close = np.shape(X_close)

    # exclude the variables that have 0 variance in the forecasted day
    X, removed_zerovar_foreday_X = remove_zerovar(X)
    X_close, removed_zerovar_foreday_X_close = remove_zerovar(X_close)

    X = X.reshape(np.shape(X)[0], np.prod(np.shape(X)[-2:])) # reshape the X to vector Days x Variables no.
    X_close = X_close.reshape(np.shape(X_close)[0], np.prod(np.shape(X_close)[-2:]))[7:,:]

    # rewrite dummies to the linear form
    linear_dummies = np.zeros((np.shape(X)[0], 7))
    for d in range(np.shape(X)[0]):
        linear_dummies[d,int(dummies_col[d])] = 1

    # stack X vector and naive forecast (last known price without differencing)
    X = np.hstack((X, np.expand_dims(daily_data_window[-len(X):,delivery_time,-(20 + forecasting_horizon)], 1)))

    # add lagged prices deltas at the time we forecast for from 1 - 7 D before
    list_of_ar = []
    for ar_idx in range(7):
        list_of_ar.append(np.expand_dims(Y[ar_idx:-7+ar_idx], 1))
    X = np.hstack([X[7:,:]] + list_of_ar)

    if preprocess_option == 0:
        scaler = StandardScaler()
    elif preprocess_option == 1:
        scaler = MinMaxScaler()

    if args.consider_exog:
        # add the last known aggregated volume information
        X = np.hstack((X, X_exog[-len(X):,:,-1]))
        # add load and gen forecasts
        Load_fore = Load[(Load.index.hour == int(delivery_time*0.25)) & (Load.index.minute == int((delivery_time*0.25 - int(delivery_time*0.25))*4*15))][1:1+idx]['Forecast']
        X = np.hstack((X, np.expand_dims(Load_fore[-len(X):],1)))
        Gen_fore = gen[(gen.index.hour == int(delivery_time*0.25)) & (gen.index.minute == int((delivery_time*0.25 - int(delivery_time*0.25))*4*15))][1:1+idx]['SPV DA'] + gen[(gen.index.hour == int(delivery_time*0.25)) & (gen.index.minute == int((delivery_time*0.25 - int(delivery_time*0.25))*4*15))][1:1+idx]['W DA']
        X = np.hstack((X, np.expand_dims(Gen_fore[-len(X):],1)))

        # add an actual DA price for the delivery time
        DA_price = DA_qtrly[(DA_qtrly.index.hour == int(delivery_time*0.25)) & (DA_qtrly.index.minute == int((delivery_time*0.25 - int(delivery_time*0.25))*4*15))][1:1+idx]['Day-ahead Price [EUR/MWh]']
        X = np.hstack((X, np.expand_dims(DA_price[-len(X):],1)))

        # add an actual ID price for the delivery time
        DA_price = ID_qtrly[(ID_qtrly.index.hour == int(delivery_time*0.25)) & (ID_qtrly.index.minute == int((delivery_time*0.25 - int(delivery_time*0.25))*4*15))][1:1+idx]['price']
        X = np.hstack((X, np.expand_dims(DA_price[-len(X):],1)))

        # add the last known actual load, load forecast error, actual gen and gen forecast error and actual DE-FR exchange
        exog_avail_mins = trade_time - forecasting_horizon # 61mins to account for data avail
        for exog_idx, exog_df in enumerate([Load, gen, ge_fr]):
            if exog_idx != 2:
                shift = 61
            else:
                shift = 121
            datetime_avail = (pd.to_datetime(date_fore) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=exog_avail_mins - shift)
            if datetime_avail.date() < date_fore.date(): # we are shifted by one day
                exog_df = exog_df[exog_df.index <= pd.to_datetime(date_fore.date())]
            else:
                exog_df = exog_df[exog_df.index <= pd.to_datetime(date_fore.date() + timedelta(days=1))]

            if exog_idx != 2:
                exog_last_info = exog_df[(exog_df.index.hour == datetime_avail.hour) & (exog_df.index.minute == datetime_avail.minute//15 * 15)]
            else:
                exog_last_info = exog_df[exog_df.index.hour == datetime_avail.hour]
            if exog_idx == 0:
                X = np.hstack((X, np.expand_dims(exog_last_info['Actual'][-len(X):],1)))
                exog_last_info_err = exog_last_info['Actual'] - exog_last_info['Forecast']
                X = np.hstack((X, np.expand_dims(exog_last_info_err[-len(X):],1)))             
            elif exog_idx == 1:
                X = np.hstack((X, np.expand_dims(exog_last_info['SPV'][-len(X):] + exog_last_info['W'][-len(X):],1)))
                exog_last_info_err = exog_last_info['SPV'] + exog_last_info['W'] - exog_last_info['SPV DA'] - exog_last_info['W DA']
                X = np.hstack((X, np.expand_dims(exog_last_info_err[-len(X):],1)))   
            elif exog_idx == 2:
                X = np.hstack((X, np.expand_dims(exog_last_info[exog_last_info.columns[0]][-len(X):],1)))
        # add the sum of volume traded for forecasted delivery in the last 60 minutes
        X = np.hstack((X, np.expand_dims(np.sum(volume_unaggr_undiff[-len(X):,-60:], 1),1)))
        # add the number of minutes with trades in them in the last 60 minutes
        X = np.hstack((X, np.expand_dims(np.sum(volume_unaggr_undiff[-len(X):,-60:] > 0, 1),1)))
        # prepare a set of exog for kernel regression
        X_exog_fundamental = X[:, -12:]
        X_exog_fundamental = np.hstack((X_exog_fundamental, dummies_col[-len(X_exog_fundamental):,np.newaxis])) # add the nonlinear dummies column

        X_exog_fundamental_plus_price = X_exog_fundamental.copy()
        X_exog_fundamental_plus_price = np.hstack((X_exog_fundamental_plus_price, np.expand_dims(daily_data_window[-len(X_exog_fundamental_plus_price):,delivery_time,-(20 + forecasting_horizon)], 1)))
        X_exog_fundamental_plus_price = np.hstack((X_exog_fundamental_plus_price, np.expand_dims(daily_data_window[-len(X_exog_fundamental_plus_price):,delivery_time,-(20 + forecasting_horizon)] - daily_data_window[-len(X_exog_fundamental_plus_price):,delivery_time,-2*(20 + forecasting_horizon)], 1)))

        X_exog_fundamental = scaler.fit_transform(X_exog_fundamental)
        X_exog_fundamental_plus_price = scaler.fit_transform(X_exog_fundamental_plus_price)

    if model != 'lasso': # add the nonlinear dummies
        X = np.hstack((X, dummies_col[-len(X):,np.newaxis]))

    X = scaler.fit_transform(X)

    # add the linearized dummies - they do not need to be standardized
    if model == 'lasso':
        X = np.hstack((X, linear_dummies[-len(X):]))
    shape_of_underlying = np.shape(X)

    shift_for_rejection = 0
    if args.no_filter_exog and model != 'lasso':
        shift_for_rejection = 21
    elif args.no_filter_exog: # 6 more dummies columns are present for LASSO due to linear dummies
        shift_for_rejection = 27

    corr = np.corrcoef(X, rowvar=False) # remove columns with correlation => threshold
    p = np.argwhere(np.triu(np.abs(corr) >= 0.8, 1))
    removed_corr = p.copy()
    p = p[p[:, 1] <= np.shape(X)[1] - shift_for_rejection]
    X = np.delete(X,p[:,1],axis=1)

    if args.save_corr_rejected_variables:
        removed_corr_df = pd.DataFrame()
        removed_corr_df['removed indices'] = removed_corr[:,1]
        removed_corr_df['correlated with'] = removed_corr[:,0]
        removed_corr_df.to_csv(f"{corr_filter_results_folname}/{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}/{calibration_flag}_{str((pd.to_datetime(date_fore) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=trade_time)).replace(':',';')}_{forecasting_horizon}_{variables_set}_weights_{weighting_alpha}_window_{window_len}_S1_removed_0_var_{removed_zerovar_foreday_X}_shape_of_underlying_{shape_of_underlying}_init_shape_{initial_shape_X}.csv")

    if np.shape(X_close)[1] > 0: # if length X_close is nonzero: transform, else copy X
        X_close = scaler.fit_transform(X_close)
        shape_of_underlying = np.shape(X_close)
        corr = np.corrcoef(X_close, rowvar=False)
        p = np.argwhere(np.triu(np.abs(corr) >= 0.95,1))
        removed_corr = p.copy()
        X_close = np.delete(X_close,p[:,1],axis=1)
        if args.save_corr_rejected_variables:
            removed_corr_df = pd.DataFrame()
            removed_corr_df['removed indices'] = removed_corr[:,1]
            removed_corr_df['correlated with'] = removed_corr[:,0]
            removed_corr_df.to_csv(f"{corr_filter_results_folname}/{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}/{calibration_flag}_{str((pd.to_datetime(date_fore) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=trade_time)).replace(':',';')}_{forecasting_horizon}_{variables_set}_weights_{weighting_alpha}_window_{window_len}_S2_removed_0_var_{removed_zerovar_foreday_X_close}_shape_of_underlying_{shape_of_underlying}_init_shape_{initial_shape_X_close}.csv")
    else:
        X_close = X.copy()

    # transform the target
    if preprocess_option == 0:
        Y_standarized = (Y[:-1] - np.mean((Y[:-1])))/np.std(Y[:-1])
    elif preprocess_option == 1:
        Y_standarized = (Y[:-1] - np.min(Y[:-1]))/(np.max(Y[:-1]) - np.min(Y[:-1]))
    Y_standarized = Y_standarized[-len(X)+1:]

    # remove 0 variance variables
    if np.shape(X[:, :np.shape(X)[1] - shift_for_rejection])[1] > 0: # might happen if we remove the 0 variance elements (for example from Sun to Mon)
        selector = VarianceThreshold().fit(X[:, :np.shape(X)[1] - shift_for_rejection])
        removed_zerovar_tot = np.where(~selector.get_support())[0]
        X[:, :np.shape(X)[1] - shift_for_rejection] = selector.transform(X[:, :np.shape(X)[1] - shift_for_rejection])

    if np.shape(X_close)[1] > 0: # might happen if we remove the 0 variance elements (for example from Sun to Mon)
        try:
            # remove 0 variance variables from close set and if it is only 0 variance substitute the close set
            selector = VarianceThreshold().fit(X_close)
            removed_zerovar_tot = np.where(~selector.get_support())[0]
            X_close = selector.transform(X_close)
            if np.shape(X_close)[1] == 0:
                X_close = X.copy()
        except:
            pass

    # apply the corr filter 2nd time if LASSO and no. of variables is > than no. of samples
    if model == 'lasso' and  np.shape(X)[1] + 14 > np.shape(X)[0]:
        no_of_variables_to_reject = np.shape(X)[1] + 14 - np.shape(X)[0]
        start_time = time.time()
        # tune the correlation threshold to reject more until no. of samples > no. of variables
        corr = np.corrcoef(X, rowvar=False)
        highly_correlated_variables = np.unravel_index(np.argsort(np.triu(np.abs(corr), 1), axis=None), np.triu(np.abs(corr), 1).shape)[1]
        highly_correlated_variables = highly_correlated_variables[highly_correlated_variables <= np.shape(X)[1] - shift_for_rejection][::-1]
        _, idx = np.unique(highly_correlated_variables, return_index=True)
        X = np.delete(X,highly_correlated_variables[np.sort(idx)][:no_of_variables_to_reject], axis=1)

    if args.model == 'lasso':
        # prepare a time series appropriate CV setting
        train_idxs = []
        test_idxs = []
        for cv_test_idx in range(14): # test on the last two weeks
            train_idxs.append(list(range(0,len(Y_standarized)-14+cv_test_idx)))
            test_idxs.append(len(Y_standarized)-14 + cv_test_idx)
        # define model and predict for correlation based variables choice from X11 variables set
        reg = LassoLarsCV(cv=zip(train_idxs, test_idxs),max_iter=1000, fit_intercept = False).fit(X[:-1,:], Y_standarized)
        pred = reg.predict(X[np.newaxis,-1,:])
        results['insample MAE'] = [my_mae(reg.predict(X[:-1,:]), Y_standarized)]
        if preprocess_option == 0:
            results[f'prediction'] = pred*np.std(Y[:-1]) + np.mean((Y[:-1])) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)
        elif preprocess_option == 1:
            results[f'prediction'] = pred*(np.max(Y[:-1]) - np.min(Y[:-1])) + np.min(Y[:-1]) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)
        # define model and predict for expert close set
        reg = LassoLarsCV(cv=zip(train_idxs, test_idxs),max_iter=1000, fit_intercept = False).fit(X_close[:-1,:], Y_standarized)
        pred = reg.predict(X_close[np.newaxis,-1,:])
        results['insample MAE close'] = [my_mae(reg.predict(X_close[:-1,:]), Y_standarized)]
        if preprocess_option == 0:
            results[f'prediction_close'] = pred*np.std(Y[:-1]) + np.mean((Y[:-1])) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)
        elif preprocess_option == 1:
            results[f'prediction_close'] = pred*(np.max(Y[:-1]) - np.min(Y[:-1])) + np.min(Y[:-1]) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)
        # define model and predict for expert exogenous set with naive variable
        reg = LassoLarsCV(cv=zip(train_idxs, test_idxs),max_iter=1000, fit_intercept = False).fit(X_exog_fundamental_plus_price[:-1,:], Y_standarized)
        pred = reg.predict(X_exog_fundamental_plus_price[np.newaxis,-1,:])
        results['insample MAE exog'] = [my_mae(reg.predict(X_exog_fundamental_plus_price[:-1,:]), Y_standarized)]
        if preprocess_option == 0:
            results[f'prediction_exog'] = pred*np.std(Y[:-1]) + np.mean((Y[:-1])) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)
        elif preprocess_option == 1:
            results[f'prediction_exog'] = pred*(np.max(Y[:-1]) - np.min(Y[:-1])) + np.min(Y[:-1]) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)

    elif model == 'random_forest':
        regr = RandomForestRegressor(n_estimators = 256, max_depth = 8) # tips on parameters: https://stats.stackexchange.com/questions/189344/choosing-random-forests-parameters
        regr.fit(X[:-1,:], Y_standarized)
        pred = regr.predict(X[np.newaxis,-1,:])
        results['insample MAE'] = [my_mae(regr.predict(X[:-1,:]), Y_standarized)]
        if preprocess_option == 0:
            results[f'prediction'] = pred*np.std(Y[:-1]) + np.mean((Y[:-1])) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)
        elif preprocess_option == 1:
            results[f'prediction'] = pred*(np.max(Y[:-1]) - np.min(Y[:-1])) + np.min(Y[:-1]) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)
        # define model and predict based on close set
        regr = RandomForestRegressor(n_estimators = 256, max_depth = 8) # tips on parameters: https://stats.stackexchange.com/questions/189344/choosing-random-forests-parameters
        regr.fit(X_close[:-1,:], Y_standarized)
        pred = regr.predict(X_close[np.newaxis,-1,:])
        results['insample MAE close'] = [my_mae(regr.predict(X_close[:-1,:]), Y_standarized)]
        if preprocess_option == 0:
            results[f'prediction_close'] = pred*np.std(Y[:-1]) + np.mean((Y[:-1])) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)
        elif preprocess_option == 1:
            results[f'prediction_close'] = pred*(np.max(Y[:-1]) - np.min(Y[:-1])) + np.min(Y[:-1]) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)
        # define model and predict for expert exogenous model with naive variable
        regr = RandomForestRegressor(n_estimators = 256, max_depth = 8) # tips on parameters: https://stats.stackexchange.com/questions/189344/choosing-random-forests-parameters
        regr.fit(X_exog_fundamental_plus_price[:-1,:], Y_standarized)
        pred = regr.predict(X_exog_fundamental_plus_price[np.newaxis,-1,:])
        results['insample MAE exog'] = [my_mae(regr.predict(X_exog_fundamental_plus_price[:-1,:]), Y_standarized)]
        if preprocess_option == 0:
            results[f'prediction_exog'] = pred*np.std(Y[:-1]) + np.mean((Y[:-1])) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)
        elif preprocess_option == 1:
            results[f'prediction_exog'] = pred*(np.max(Y[:-1]) - np.min(Y[:-1])) + np.min(Y[:-1]) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)

    elif model == 'dense':
        batch_size = 16
        inputs = keras.Input(X.shape[1:])
        # flatten
        flat = keras.layers.Flatten()(inputs)
        # batch normalization
        norm = keras.layers.BatchNormalization()(flat)
        last_layer = norm
        # dropout
        # rate = 0.05 # trial.suggest_float('dropout_rate', 0, 1)
        # drop = keras.layers.Dropout(rate)(last_layer)
        # last_layer = drop
        # regularization of 1st hidden layer,
        # activation - output, kernel - weights/parameters of input
        # define 1st hidden layer with regularization
        hidden = keras.layers.Dense(128,
                                    activation='relu')(last_layer)
        # regularization of 2nd hidden layer,
        # activation - output, kernel - weights/parameters of input
        # define 2nd hidden layer with regularization
        # hidden = keras.layers.Dense(512,
        #                             activation='relu')(hidden)
        # define 3rd hidden layer with regularization
        # hidden = keras.layers.Dense(1024,
        #                             activation='relu',
        #                             # kernel_initializer='ones',
        #                             kernel_regularizer=keras.regularizers.L1(h2_kernel_rate),
        #                             activity_regularizer=keras.regularizers.L1(h2_activation_rate))(hidden)
        hidden = keras.layers.Dense(64,
                                    activation='relu')(hidden)
        outputs = keras.layers.Dense(1, activation='linear')(hidden)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=1e-2,
                        decay_steps=4*(idx//batch_size),
                        decay_rate=0.9,
                        staircase=True)
        print(f'Decay every {4*(idx//batch_size)} steps.')

        n_samples = np.shape(X)[0] - 1
        all_idxs = np.arange(n_samples)
        trainsubset = all_idxs[:int(n_samples*4/5)]
        valsubset = all_idxs[int(n_samples*4/5):]

        callbacks = [keras.callbacks.EarlyStopping(patience = 32, restore_best_weights=True)]
        preds = []
        for _ in range(5):
            keras_model = keras.Model(inputs=inputs, outputs=outputs)
            keras_model.compile(optimizer=keras.optimizers.Adam(lr_schedule),
                        loss='mse',
                        metrics='mse')
            history = keras_model.fit(X[:-1,:][trainsubset,:], Y_standarized[trainsubset], epochs=64, validation_data=(X[:-1,:][valsubset,:], Y_standarized[valsubset]), callbacks=callbacks, batch_size=batch_size, verbose=1)
            # summarize history for accuracy
            # plt.figure()
            # plt.plot(history.history['mse'])
            # plt.plot(history.history['val_mse'])
            # plt.title('Model MSE')
            # plt.ylabel('MSE')
            # plt.xlabel('epoch')
            # plt.legend(['train', 'test'], loc='upper left')
            # plt.show()

            pred = keras_model.predict(X[np.newaxis,-1])[0][0] # add a new axis to fit into the shape demanded by the model
            preds.append(pred)
        pred = np.mean(preds)

    elif model == 'kernel_hr_naive_mult':

        training_window = X[:-1,:]
        training_window_close = X_close[:-1,:]
        if args.consider_exog:
            training_window_fundamental = X_exog_fundamental[:-1,:]
            training_window_fundamental_plus_price = X_exog_fundamental_plus_price[:-1,:]
        naive_vec = daily_data_window[7:,delivery_time,-(20 + forecasting_horizon)]

        if preprocess_option == 0:
            naive_vec_standardized = (naive_vec - np.mean(naive_vec))/np.std(naive_vec)
        elif preprocess_option == 1:
            naive_vec_standardized = (naive_vec - np.min(naive_vec))/(np.max(naive_vec) - np.min(naive_vec))

        @njit
        def calc_interm_kernel(interm_data, norm: int = 2):
            plain_kernel = np.zeros((len(interm_data), len(interm_data)))
            for i in range(len(interm_data)):
                for j in range(len(interm_data)):
                    if i < j:
                        if norm == 2:
                            plain_kernel[i, j] = np.sqrt(np.sum((interm_data[i] - interm_data[j])**2)) # L2 distance
                        elif norm == 1:
                            plain_kernel[i, j] = np.sum(np.abs(interm_data[i] - interm_data[j])) # L1 distance
            plain_kernel = plain_kernel + plain_kernel.T
            return(plain_kernel)

        if args.consider_exog:
            plain_kernel_exog_train = calc_interm_kernel(training_window_fundamental)
            plain_kernel_exog_test = np.sqrt(np.sum((training_window_fundamental - X_exog_fundamental[np.newaxis,-1,:])**2, axis=1))

            plain_kernel_exog_plus_prices_train = calc_interm_kernel(training_window_fundamental_plus_price)
            plain_kernel_exog_plus_prices_test = np.sqrt(np.sum((training_window_fundamental_plus_price - X_exog_fundamental_plus_price[np.newaxis,-1,:])**2, axis=1))

        else:
            plain_kernel_exog_train = None
            plain_kernel_exog_test = None
            plain_kernel_exog_plus_prices_train = None
            plain_kernel_exog_plus_prices_test = None

        plain_kernel_X_1_train = calc_interm_kernel(training_window)

        plain_kernel_close_train = calc_interm_kernel(training_window_close)

        plain_kernel_X_1_test = np.sqrt(np.sum((training_window - X[np.newaxis,-1,:])**2, axis=1))

        plain_kernel_close_test = np.sqrt(np.sum((training_window_close - X_close[np.newaxis,-1,:])**2, axis=1))

        plain_kernel_X_2_test = np.abs(naive_vec_standardized[:-1]  - naive_vec_standardized[np.newaxis,-1])**2 # in 1D the Euclidean distance is just an abs value raised to power 2

        plain_kernel_X_2_train = np.abs(np.tile(naive_vec_standardized[:-1], [len(naive_vec_standardized[:-1]), 1]) - np.swapaxes(np.tile(naive_vec_standardized[:-1], [len(naive_vec_standardized[:-1]), 1]), 0, 1))**2

        q_kernel = 0.75 # WARNING only for q >= 1/2
        q_kernel_naive = 0.75 # WARNING only for q >= 1/2
        q_data = 0.5
        q_data_naive = 0.75

        kernels_train = [ 
                        plain_kernel_X_1_train,
                        plain_kernel_close_train,
                        plain_kernel_exog_plus_prices_train,
                        plain_kernel_X_2_train
                    ]

        kernels_test = [ 
                        plain_kernel_X_1_test,
                        plain_kernel_close_test,
                        plain_kernel_exog_plus_prices_test,
                        plain_kernel_X_2_test
                    ]

        # L1 S_1 kernels for laplace SVR input to compare with L2
        plain_kernel_X_1_train_L1 = calc_interm_kernel(training_window, norm=1)

        plain_kernel_X_1_test_L1 = np.sum(np.abs(training_window - X[np.newaxis,-1,:]), axis=1)

        def fast_calculate_kernel_matrix(stage, kernel_option = 0):

            sigma = np.quantile(kernels_train[-1], q_data_naive) / scipy.stats.norm.ppf(q_kernel_naive, loc=0, scale=1) # parameters are ALWAYS based on the train window so thet we use the same kernel 

            if stage == 'train':
                calc_kernels = kernels_train
                calc_kernel_L1 = plain_kernel_X_1_train_L1
            elif stage == 'test':
                calc_kernels = kernels_test
                calc_kernel_L1 = plain_kernel_X_1_test_L1

            if kernel_option == 1:
                width = np.log(2 - 2*q_kernel)/np.quantile(kernels_train[0], q_data)
                return np.exp( width * calc_kernels[0] - 1/(2*sigma**2) * calc_kernels[-1])
            elif kernel_option == 2:
                width = np.log(2 - 2*q_kernel)/np.quantile(kernels_train[1], q_data)
                return np.exp( width * calc_kernels[1] - 1/(2*sigma**2) * calc_kernels[-1])
            elif kernel_option == 7: # exog with added info on naive and differenced naive
                width = np.log(2 - 2*q_kernel)/np.quantile(kernels_train[2], q_data)
                return np.exp( width * calc_kernels[2] - 1/(2*sigma**2) * calc_kernels[-1])
            elif kernel_option == 'plain_laplace_L2':
                width = np.log(2 - 2*q_kernel)/np.quantile(kernels_train[0], q_data)
                return np.exp( width * calc_kernels[0])
            elif kernel_option == 'plain_laplace':
                width = np.log(2 - 2*q_kernel)/np.quantile(plain_kernel_X_1_train_L1, q_data)
                return np.exp( width * calc_kernel_L1)

        # to check whether it is positive semidefinite use: https://stackoverflow.com/questions/16266720/find-out-if-a-matrix-is-positive-definite-with-numpy
        for kernel_choice in [1,2,7,'plain_laplace','plain_laplace_L2']:
            training_matrix = fast_calculate_kernel_matrix(stage='train', kernel_option = kernel_choice)
            if kernel_model == 'KRR':
                estimator = KernelRidge(kernel = 'precomputed', alpha = alpha_KRR)
            elif kernel_model == 'SVR':
                estimator = SVR(kernel = 'precomputed', epsilon = svr_epsilon, C = C)
            estimator.fit(training_matrix, Y_standarized[:,np.newaxis])
            results[f'insample MAE_{kernel_choice}'] = [my_mae(estimator.predict(training_matrix), Y_standarized)]
            test_matrix = fast_calculate_kernel_matrix(stage='test', kernel_option = kernel_choice)
            pred = estimator.predict(test_matrix.reshape(1,-1))
            if preprocess_option == 0:
                results[f'prediction_{kernel_choice}'] = pred*np.std(Y[:-1]) + np.mean((Y[:-1])) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)
            elif preprocess_option == 1:
                results[f'prediction_{kernel_choice}'] = pred*(np.max(Y[:-1]) - np.min(Y[:-1])) + np.min(Y[:-1]) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)

    results['actual'] = [daily_data_window[-1,delivery_time,-1]]
    results['naive'] = [daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)]]

    if model not in ['kernel_hr_naive_mult', 'lasso', 'random_forest']: # kernel model has already stored its results
        if preprocess_option == 0:
            results['prediction'] = pred*np.std(Y[:-1]) + np.mean((Y[:-1])) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)
        elif preprocess_option == 1:
            results['prediction'] = pred*(np.max(Y[:-1]) - np.min(Y[:-1])) + np.min(Y[:-1]) + daily_data_window[-1,delivery_time,-(20 + forecasting_horizon)] # prediction is the sum of forecasted signal and last known price (so we think of it as fine tuning the naive forecast)
    try:
        results.to_csv(f"{results_folname}/{model}_{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}/{calibration_flag}_{str((pd.to_datetime(date_fore) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=trade_time)).replace(':',';')}_{forecasting_horizon}_{variables_set}_weights_{weighting_alpha}_window_{window_len}.csv")
    except:
        os.remove(f"{results_folname}/{model}_{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}/{calibration_flag}_{str((pd.to_datetime(date_fore) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=trade_time)).replace(':',';')}_{forecasting_horizon}_{variables_set}_weights_{weighting_alpha}_window_{window_len}.csv")
        raise KeyboardInterrupt("Interrupted on saving: last file removed to avoid empty files.")

if __name__ == '__main__':
    con = sqlite3.connect('data_ID.db')
    sql_str = f"SELECT * FROM with_dummies WHERE Index_daily <= {trade_time};" # load only the data required for simu, so up to trade time
    daily_data = pd.read_sql(sql_str, con)[[str(i) for i in range(193)]].to_numpy()
    end = time.time()
    daily_data = np.reshape(daily_data, (np.shape(daily_data)[0]//trade_time, trade_time, 193))

    # scan the directory for existing files
    print("Checking for simulation results that are already saved...")
    run_simu = False
    required_variables_sets = []
    required_forecasting_horizons = []
    required_dates = {}
    required_indices = {}
    required_dates_calibration = {}
    required_indices_calibration = {}
    for forecasting_horizon in forecasting_horizons:
        required_dates[forecasting_horizon] = {}
        required_indices[forecasting_horizon] = {}
        required_dates_calibration[forecasting_horizon] = {}
        required_indices_calibration[forecasting_horizon] = {}
        if np.shape(daily_data)[1] <= 2*(20 + forecasting_horizon):
            print((f"Differencing could not be performed with series of length {np.shape(daily_data)[1]}min and horizon {forecasting_horizon}min (+20min of data avail delay.)"))
            continue
        for variables_set in [variable_set]:
            required_dates[forecasting_horizon][variables_set] = []
            required_indices[forecasting_horizon][variables_set] = []
            required_dates_calibration[forecasting_horizon][variables_set] = []
            required_indices_calibration[forecasting_horizon][variables_set] = []
            for date_idx, date in enumerate(dates):
                if not os.path.isfile(f"{results_folname}/{model}_{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}/test_{str((pd.to_datetime(date) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=trade_time)).replace(':',';')}_{forecasting_horizon}_{variables_set}_weights_{weighting_alpha}_window_{window_len}.csv"):
                    run_simu = True
                    if variables_set not in required_variables_sets:
                        required_variables_sets.append(variables_set)
                    if forecasting_horizon not in required_forecasting_horizons:
                        required_forecasting_horizons.append(forecasting_horizon)
                    required_dates[forecasting_horizon][variables_set].append(date)
                    required_indices[forecasting_horizon][variables_set].append(date_idx)
            for date_idx, date in enumerate(dates_calibration):
                if not os.path.isfile(f"{results_folname}/{model}_{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}/calibration_{str((pd.to_datetime(date) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=trade_time)).replace(':',';')}_{forecasting_horizon}_{variables_set}_weights_{weighting_alpha}_window_{window_len}.csv"):
                    required_dates_calibration[forecasting_horizon][variables_set].append(date)
                    required_indices_calibration[forecasting_horizon][variables_set].append(date_idx)
            if run_simu:
                print(f"Config fore. horizon: {forecasting_horizon}, variable set: {variables_set} needs a recalculation.")
    if not run_simu:
        print("All required config was already simulated.")
        sys.exit(0)

    raw_arr = RawArray('d', np.shape(daily_data)[0] * np.shape(daily_data)[1] * np.shape(daily_data)[2])

    # Wrap X as an numpy array so we can easily manipulates its data in multiprocessing scheme
    daily_data_np = np.frombuffer(raw_arr).reshape(np.shape(daily_data))
    # Copy data to our shared array.
    np.copyto(daily_data_np, daily_data)
    data_shape = np.shape(daily_data)

    # free memory
    daily_data = None
    del daily_data

    simu_start = time.time()
    for forecasting_horizon in required_forecasting_horizons:
        for variables_set in required_variables_sets:
            if calibration_window_len > 0: # perform the calibration forecast
                inputlist_calibration = [[lookback - calibration_window_len + idx, date, forecasting_horizon, variables_set, model, 'calibration'] for idx, date in zip(required_indices_calibration[forecasting_horizon][variables_set], required_dates_calibration[forecasting_horizon][variables_set])]
                try:
                    with Pool(processes=int(args.processes), initializer=init_worker, initargs=(raw_arr, data_shape)) as p:
                        _ = p.map(run_one_day, inputlist_calibration)
                except Exception as exception:
                    print(f"Failed pool due to {exception}. Restarting with 15 workers")
                    with Pool(processes=15, initializer=init_worker, initargs=(raw_arr, data_shape)) as p:
                        _ = p.map(run_one_day, inputlist_calibration)

                inputlist = [[lookback + idx, date, forecasting_horizon,variables_set,model,'test'] for idx, date in zip(required_indices[forecasting_horizon][variables_set], required_dates[forecasting_horizon][variables_set])]
                try:
                    with Pool(processes=int(args.processes), initializer=init_worker, initargs=(raw_arr, data_shape)) as p:
                        _ = p.map(run_one_day, inputlist)
                except Exception as exception:
                    print(f"Failed pool due to {exception}. Restarting with 15 workers")
                    with Pool(processes=15, initializer=init_worker, initargs=(raw_arr, data_shape)) as p:
                        _ = p.map(run_one_day, inputlist)

            else:
                inputlist = [[lookback + idx, date, forecasting_horizon,variables_set,model,'test'] for idx, date in zip(required_indices[forecasting_horizon][variables_set], required_dates[forecasting_horizon][variables_set])]
                with Pool(processes=int(args.processes), initializer=init_worker, initargs=(raw_arr, data_shape)) as p:
                    _ = p.map(run_one_day, inputlist)
    end = time.time()
    print(end - simu_start, 'Total time of simulation:')
    with open(f"timing_results_model_{model}_d_{delivery_time}_t_{trade_time}.txt", "w") as file:
        file.write(f"Execution time: {end - simu_start} seconds\n")
