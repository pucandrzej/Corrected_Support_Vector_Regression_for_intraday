import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import timedelta

def plot_avg_day_trade(df_copy, prices, counter, 
                       weekdays, dates, title='', week_days_chosen=range(7), 
                       ax=None, cbar_label='', save_title='3D_analysis',
                       xlabel=False, ylabel=False, return_min_max=False, cbar_limits=None):
    """Function for averaging and plotting prices statistics.

    Args:
        df_copy (pandas data frame): original data frame
        prices (numpy array): prices statistics
        counter (numpy array): array of counters (numbers of trades for every time bin)
        weekdays (numpy array): weekdays of dates
        dates (DatetimeIndex): daterange of dates
        title (str): title of the plot where 1st word of it is name of statistic to be averaged and the last - 1 word is the current/next word (to specify whether the passed statistics are for trade on the next or current day.
    """
    dates_day = pd.date_range(start=np.min(df_copy["Datetime offer time"]), end=np.max(
        df_copy["Datetime offer time"]), freq='D', normalize=True)
    weekdays_day = []
    for d in dates_day:
        weekdays_day.append(d.weekday())
    weekdays_day = np.array(weekdays_day)
    weekd = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for wd in week_days_chosen:
        # for the averaging purpose
        daily_array_1 = np.zeros((np.shape(prices)[0], 24))
        daily_array_1[:] = np.nan
        daily_counter_1 = np.zeros((np.shape(prices)[0], 24))
        daily_counter_1[:] = np.nan
        prices_wday = prices[:, weekdays == wd]
        counter_wday = counter[:, weekdays == wd]
        dates_wday = dates[weekdays == wd]
        dates_day_wday = dates_day[weekdays_day == wd]
        # return prices_wday
        for dday in dates_day_wday:
            slots_day_offer = pd.date_range(
                start=dday, end=dday + timedelta(1), freq='H', inclusive='left')
            for idx0, dminute in enumerate(dates_wday):
                if dday == dminute.normalize():
                    idx = np.where(slots_day_offer == dminute)[0][0]
                    daily_array_1[:, idx] = np.nansum(
                        np.dstack((prices_wday[:, idx0], daily_array_1[:, idx])), 2)
                    daily_counter_1[:, idx] = np.nansum(
                        np.dstack((counter_wday[:, idx0] > 0, daily_counter_1[:, idx])), 2)
        daily_counter_1[daily_counter_1 == 0] = np.nan
        daily_array_1 = daily_array_1/daily_counter_1

        if return_min_max:
            return (np.nanmin(daily_array_1), np.nanmax(daily_array_1))

        if ax == None:
            fig, ax = plt.subplots(figsize=(5, 5), sharex=True, sharey=True)
            im = ax.imshow(daily_array_1, origin='lower', aspect='auto',
                        cmap='turbo',
                        extent=[0, 24, 0, 96])
            plt.gcf().autofmt_xdate()
            cbar = fig.colorbar(im)
            cbar.set_label(cbar_label, rotation=270, labelpad=12)
            plt.xlabel('Offer time [h]')
            plt.grid()
            plt.ylabel('Delivery time [quarter no. of day]')
            ax.set_title(f'{title} {weekd[wd]}')
            plt.savefig(
                f'Paper_Figures/{save_title}-{weekd[wd]}.pdf')
            plt.show()
        else:
            if cbar_limits != None:
                im = ax.imshow(daily_array_1, origin='lower', aspect='auto',
                        cmap='turbo',
                        extent=[0, 24, 0, 96],
                        vmin=cbar_limits[0], vmax=cbar_limits[1])
            else:
                im = ax.imshow(daily_array_1, origin='lower', aspect='auto',
                        cmap='turbo',
                        extent=[0, 24, 0, 96])
            plt.gcf().autofmt_xdate()
            cbar = plt.gcf().colorbar(im)
            cbar.set_label(cbar_label, rotation=270, labelpad=12)
            if xlabel:
                ax.set_xlabel('Offer time [h]')
            ax.grid(visible=True)
            if ylabel:
                ax.set_ylabel('Delivery quarter')
            ax.set_title(f'{title} {weekd[wd]}')

def plot_avg_day_trade_glued(df_copy, prices_1, counter_1, prices_2, counter_2, 
                       weekdays, dates, title='', week_days_chosen=range(7), 
                       ax=None, cbar_label='', save_title='3D_analysis',
                       xlabel=False, ylabel=False, return_min_max=False, cbar_limits=None):
    """Function for averaging and plotting prices_1 statistics.

    Args:
        df_copy (pandas data frame): original data frame
        prices_1 (numpy array): prices_1 statistics
        counter_1 (numpy array): array of counters (numbers of trades for every time bin)
        weekdays (numpy array): weekdays of dates
        dates (DatetimeIndex): daterange of dates
        title (str): title of the plot where 1st word of it is name of statistic to be averaged and the last - 1 word is the current/next word (to specify whether the passed statistics are for trade on the next or current day.
    """
    dates_day = pd.date_range(start=np.min(df_copy["Datetime offer time"]), end=np.max(
        df_copy["Datetime offer time"]), freq='D', normalize=True)
    weekdays_day = []
    for d in dates_day:
        weekdays_day.append(d.weekday())
    weekdays_day = np.array(weekdays_day)
    weekd = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for wd in week_days_chosen:
        # for the averaging purpose
        daily_array_1 = np.zeros((np.shape(prices_1)[0], 24))
        daily_array_1[:] = np.nan
        daily_counter_1 = np.zeros((np.shape(prices_1)[0], 24))
        daily_counter_1[:] = np.nan
        prices_wday = prices_1[:, weekdays == wd]
        counter_wday = counter_1[:, weekdays == wd]
        dates_wday = dates[weekdays == wd]
        dates_day_wday = dates_day[weekdays_day == wd]
        # return prices_wday
        for dday in dates_day_wday:
            slots_day_offer = pd.date_range(
                start=dday, end=dday + timedelta(1), freq='H', inclusive='left')
            for idx0, dminute in enumerate(dates_wday):
                if dday == dminute.normalize():
                    idx = np.where(slots_day_offer == dminute)[0][0]
                    daily_array_1[:, idx] = np.nansum(
                        np.dstack((prices_wday[:, idx0], daily_array_1[:, idx])), 2)
                    daily_counter_1[:, idx] = np.nansum(
                        np.dstack((counter_wday[:, idx0] > 0, daily_counter_1[:, idx])), 2)
        daily_counter_1[daily_counter_1 == 0] = np.nan
        daily_array_1 = daily_array_1/daily_counter_1

        # for the averaging purpose
        weekdays = weekdays[:-24]
        dates = dates[:-24]
        daily_array_2 = np.zeros((np.shape(prices_2)[0], 24))
        daily_array_2[:] = np.nan
        daily_counter_2 = np.zeros((np.shape(prices_2)[0], 24))
        daily_counter_2[:] = np.nan
        print(np.shape(prices_2), np.shape(weekdays))
        prices_wday = prices_2[:, weekdays == wd]
        counter_wday = counter_2[:, weekdays == wd]
        dates_wday = dates[weekdays == wd]
        dates_day_wday = dates_day[weekdays_day == wd]
        # return prices_wday
        for dday in dates_day_wday:
            slots_day_offer = pd.date_range(
                start=dday, end=dday + timedelta(1), freq='H', inclusive='left')
            for idx0, dminute in enumerate(dates_wday):
                if dday == dminute.normalize():
                    idx = np.where(slots_day_offer == dminute)[0][0]
                    daily_array_2[:, idx] = np.nansum(
                        np.dstack((prices_wday[:, idx0], daily_array_2[:, idx])), 2)
                    daily_counter_2[:, idx] = np.nansum(
                        np.dstack((counter_wday[:, idx0] > 0, daily_counter_2[:, idx])), 2)
        daily_counter_2[daily_counter_2 == 0] = np.nan
        daily_array_2 = daily_array_2/daily_counter_2

        if return_min_max:
            return (np.nanmin([np.nanmin(daily_array_1), np.nanmin(daily_array_2)]), np.nanmax([np.nanmax(daily_array_1), np.nanmax(daily_array_2)]))
        
        concatenated_array = np.concatenate((daily_array_2, daily_array_1), axis=1)

        if ax == None:
            fig, ax = plt.subplots(figsize=(5, 5), sharex=True, sharey=True)
            im = ax.imshow(concatenated_array, origin='lower', aspect='auto',
                        cmap='turbo',
                        extent=[0, 48, 0, 96])
            plt.gcf().autofmt_xdate()
            cbar = fig.colorbar(im)
            cbar.set_label(cbar_label, rotation=270, labelpad=12)
            plt.xlabel('Offer time [h]')
            plt.grid()
            plt.ylabel('Delivery time [quarter no. of day]')
            ax.set_title(f'{title} {weekd[wd]}')
            plt.savefig(
                f'Paper_Figures/{save_title}-{weekd[wd]}.pdf')
            plt.show()
        else:
            if cbar_limits != None:
                im = ax.imshow(concatenated_array, origin='lower', aspect='auto',
                        cmap='turbo',
                        extent=[0, 48, 0, 96],
                        vmin=cbar_limits[0], vmax=cbar_limits[1])
            else:
                im = ax.imshow(concatenated_array, origin='lower', aspect='auto',
                        cmap='turbo',
                        extent=[0, 48, 0, 96])
            plt.gcf().autofmt_xdate()
            cbar = plt.gcf().colorbar(im)
            cbar.set_label(cbar_label, rotation=270, labelpad=12)
            if xlabel:
                ax.set_xlabel('Offer time [h]')
            ax.grid(visible=True)
            if ylabel:
                ax.set_ylabel('Delivery quarter')
            ax.set_title(f'{title} {weekd[wd]}')

def plot_avg_day_trade_glued_counter(df_copy, prices_1, counter_1, prices_2, counter_2, 
                       weekdays, dates, title='', week_days_chosen=range(7), 
                       ax=None, cbar_label='', save_title='3D_analysis',
                       xlabel=False, ylabel=False, return_min_max=False, cbar_limits=None):
    """Function for averaging and plotting prices_1 statistics.

    Args:
        df_copy (pandas data frame): original data frame
        prices_1 (numpy array): prices_1 statistics
        counter_1 (numpy array): array of counters (numbers of trades for every time bin)
        weekdays (numpy array): weekdays of dates
        dates (DatetimeIndex): daterange of dates
        title (str): title of the plot where 1st word of it is name of statistic to be averaged and the last - 1 word is the current/next word (to specify whether the passed statistics are for trade on the next or current day.
    """
    dates_day = pd.date_range(start=np.min(df_copy["Datetime offer time"]), end=np.max(
        df_copy["Datetime offer time"]), freq='D', normalize=True)
    weekdays_day = []
    for d in dates_day:
        weekdays_day.append(d.weekday())
    weekdays_day = np.array(weekdays_day)
    weekd = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for wd in week_days_chosen:
        # for the averaging purpose
        daily_array_1 = np.zeros((np.shape(prices_1)[0], 24))
        daily_array_1[:] = np.nan
        daily_counter_1 = np.zeros((np.shape(prices_1)[0], 24))
        daily_counter_1[:] = np.nan
        prices_wday = prices_1[:, weekdays == wd]
        counter_wday = counter_1[:, weekdays == wd]
        dates_wday = dates[weekdays == wd]
        dates_day_wday = dates_day[weekdays_day == wd]
        # return prices_wday
        for dday in dates_day_wday:
            slots_day_offer = pd.date_range(
                start=dday, end=dday + timedelta(1), freq='H', inclusive='left')
            for idx0, dminute in enumerate(dates_wday):
                if dday == dminute.normalize():
                    idx = np.where(slots_day_offer == dminute)[0][0]
                    daily_array_1[:, idx] = np.nansum(
                        np.dstack((prices_wday[:, idx0], daily_array_1[:, idx])), 2)
                    daily_counter_1[:, idx] = np.nansum(
                        np.dstack((counter_wday[:, idx0] > 0, daily_counter_1[:, idx])), 2)
        daily_counter_1[daily_counter_1 == 0] = np.nan
        daily_array_1 = daily_counter_1

        # for the averaging purpose
        weekdays = weekdays[:-24]
        dates = dates[:-24]
        daily_array_2 = np.zeros((np.shape(prices_2)[0], 24))
        daily_array_2[:] = np.nan
        daily_counter_2 = np.zeros((np.shape(prices_2)[0], 24))
        daily_counter_2[:] = np.nan
        prices_wday = prices_2[:, weekdays == wd]
        counter_wday = counter_2[:, weekdays == wd]
        dates_wday = dates[weekdays == wd]
        dates_day_wday = dates_day[weekdays_day == wd]
        # return prices_wday
        for dday in dates_day_wday:
            slots_day_offer = pd.date_range(
                start=dday, end=dday + timedelta(1), freq='H', inclusive='left')
            for idx0, dminute in enumerate(dates_wday):
                if dday == dminute.normalize():
                    idx = np.where(slots_day_offer == dminute)[0][0]
                    daily_array_2[:, idx] = np.nansum(
                        np.dstack((prices_wday[:, idx0], daily_array_2[:, idx])), 2)
                    daily_counter_2[:, idx] = np.nansum(
                        np.dstack((counter_wday[:, idx0] > 0, daily_counter_2[:, idx])), 2)
        daily_counter_2[daily_counter_2 == 0] = np.nan
        daily_array_2 = daily_counter_2

        if return_min_max:
            return (np.nanmin([np.nanmin(daily_array_1), np.nanmin(daily_array_2)]), np.nanmax([np.nanmax(daily_array_1), np.nanmax(daily_array_2)]))
        
        concatenated_array = np.concatenate((daily_array_2, daily_array_1), axis=1)

        if ax == None:
            fig, ax = plt.subplots(figsize=(5, 5), sharex=True, sharey=True)
            im = ax.imshow(concatenated_array, origin='lower', aspect='auto',
                        cmap='turbo',
                        extent=[0, 48, 0, 96])
            plt.gcf().autofmt_xdate()
            cbar = fig.colorbar(im)
            cbar.set_label(cbar_label, rotation=270, labelpad=12)
            plt.xlabel('Offer time [h]')
            plt.grid()
            plt.ylabel('Delivery time [quarter no. of day]')
            ax.set_title(f'{title} {weekd[wd]}')
            plt.savefig(
                f'Paper_Figures/{save_title}-{weekd[wd]}.pdf')
            plt.show()
        else:
            if cbar_limits != None:
                im = ax.imshow(concatenated_array, origin='lower', aspect='auto',
                        cmap='turbo',
                        extent=[0, 48, 0, 96],
                        vmin=cbar_limits[0], vmax=cbar_limits[1])
            else:
                im = ax.imshow(concatenated_array, origin='lower', aspect='auto',
                        cmap='turbo',
                        extent=[0, 48, 0, 96])
            plt.gcf().autofmt_xdate()
            cbar = plt.gcf().colorbar(im)
            cbar.set_label(cbar_label, rotation=270, labelpad=12)
            if xlabel:
                ax.set_xlabel('Offer time [h]')
            ax.grid(visible=True)
            if ylabel:
                ax.set_ylabel('Delivery quarter')
            ax.set_title(f'{title} {weekd[wd]}')

