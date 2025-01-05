'''
Script to run the simulation for the required configuration of distances before trading and forecast and different deliveries
'''

import time
import subprocess
import gc

import sys
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--start_delivery', default=0, help='Start of the simulated deliveries')
parser.add_argument('--end_delivery', default=96, help='End of the simulated deliveries')
parser.add_argument('--models', default=['kernel_hr_naive_mult', 'lasso', 'random_forest'], help='Models to simulate.')
parser.add_argument('--calibration_window_len', default=28, help='For every date consider a historical results from a calibration window.')
parser.add_argument('--kernel_solver', default='SVR', help='Solving regression based on the kernel input.')
args = parser.parse_args()
for model in args.models:
    start = args.start_delivery
    joblist = []
    sys.stderr = open(f'TOTAL_SIMU_ERR_{start}_{args.end_delivery}_{model}_{args.calibration_window_len}_{args.kernel_solver}.txt', 'w')
    sys.stdout = open(f'TOTAL_SIMU_LOG_{start}_{args.end_delivery}_{model}_{args.calibration_window_len}_{args.kernel_solver}.txt', 'w')
    for shift_trade in [30, 90, 180]: # delivery time - shift_trade is the trade time
        deliveries_recalc = [0, 48, 95]

        for delivery_time in deliveries_recalc:
            trade_time = delivery_time*15 + 8*60 - shift_trade
            for variable_set in [11]:
                processes = 30
                joblist.append(['C:/Users/riczi/Studies/Continuous_market_analysis/contmarket311/Scripts/python.exe', 'lasso_forecasting.py', '--model', model, '--trade_time', str(trade_time), '--delivery_time', str(delivery_time), '--variable_set', str(variable_set), '--kernel_solver', args.kernel_solver, '--calibration_window_len', str(args.calibration_window_len), '--processes', str(processes)])

    invoked = 0
    stack = []
    ts = time.time()
    concurrent = 1
    while invoked < len(joblist):
        while len(stack) == concurrent:
            for no, p in enumerate(stack):
                if p.poll() is not None:
                    stack.pop(no)
                    break
            time.sleep(1)
        line = joblist[invoked]
        print(f'running job {invoked+1} of {len(joblist)}: {joblist[invoked]}')
        stack.append(subprocess.Popen(line, stderr=sys.stderr, stdout=sys.stdout))
        stack[-1].wait() # wait for the process to finish
        invoked += 1
