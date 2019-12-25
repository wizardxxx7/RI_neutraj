import os
import time
import json
import math
import urllib
import folium
import datetime
import argparse
import logging
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import colors as mcolors
from matplotlib import dates as mdate
from shapely.geometry import LineString, Point
from matplotlib.backends.backend_pdf import PdfPages
from math import radians, cos, sin, asin, sqrt, degrees, atan2

COLORS = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
COLOR_LST = [k for k, v in COLORS.items() if 'dark' in k]
MARK_COLOR = [
    'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue',
    'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen',
    'gray', 'black', 'lightgray'
]


def get_utils_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-dir", type=str, help="Data path")
    args = parser.parse_args()
    return args


class RawSignal:
    def __init__(self, signal_dir):
        self.file_dir = signal_dir

    def get_signals(self, file_date, part):
        filename = os.path.join(self.file_dir, 'hf_' + str(file_date),
                                'part-' + part)
        print('Get raw signal:', filename)
        f = open(filename)
        dates, cell_id, user_id, service_type, web = [], [], [], [], []

        for line in f.readlines():
            line_tmp = line.strip('()\n').split(',')
            dates.append(get_date_type(line_tmp[0]))
            cell_id.append(line_tmp[1])
            user_id.append(line_tmp[2])
            service_type.append(line_tmp[3])
            web.append(line_tmp[4])
        f.close()
        return pd.DataFrame({
            'dates': dates,
            'cell_id': cell_id,
            'user_id': user_id,
            'service_type': service_type,
            'web': web
        })

    def time_based_signal_data(self, file_date):
        '''
        Prepare time based signal data
        :param signal_dir: (str) e.g: "~/hf_signals"
        :param file_date: (str) e.g: "20170607"
        :return:
        '''
        data_dir = os.path.join(self.file_dir, 'time_based_signals')
        check_dir(data_dir)
        out_dir = os.path.join(data_dir, file_date)
        print('Generating time based signals:', out_dir)
        if os.path.exists(out_dir):
            print('%s time based signal data already exists. Skipping ...' %
                  file_date)
            return
        if not os.path.exists(os.path.join(self.file_dir, 'hf_' + file_date)):
            print('%s signal data not exists. Skipping ...' % file_date)
            return
        check_dirs([os.path.basename(out_dir), out_dir])

        parts = sorted([
            f.split('-')[1]
            for f in os.listdir(os.path.join(self.file_dir, 'hf_' + file_date))
            if 'part' in f
        ])
        df = None
        t0 = time.time()

        for i, part in enumerate(parts):
            tmp = self.get_signals(file_date, part)
            if df is None:
                df = tmp
            else:
                df = pd.concat([df, tmp])

            if i % 30 == 0:
                df['time_minute'] = [
                    '%s%.2d' % (t.strftime('%Y%m%d-%H'), (t.minute // 5) * 5)
                    for t in df.dates
                ]
                unique_minute = np.unique(df.time_minute)
                for t_min in unique_minute:
                    out_f = os.path.join(self.file_dir, 'time_based_signals',
                                         file_date, t_min + '.csv')
                    df_min = df[df.time_minute == t_min]
                    df_min = df_min.drop(['time_minute'], axis=1)
                    if os.path.exists(out_f):
                        df_min.to_csv(out_f,
                                      index=False,
                                      mode='a',
                                      header=False)
                    else:
                        df_min.to_csv(out_f, index=False)
                df = None

            if i % 10 == 0:
                print('%s-%d parts is finished. Cost %.2f s.' %
                      (file_date, i, time.time() - t0))
                t0 = time.time()
        successed = Json.load(os.path.join(data_dir, 'successed.json'))
        successed['success'].append(file_date)
        Json.output(successed, os.path.join(data_dir, 'successed.json'))
        return

    def time_based_signal_data_async(self, file_dates, process_i=0):
        for file_date in file_dates:
            print('[Process %d] Getting time based signal for date %s' %
                  (process_i, file_date))
            self.time_based_signal_data(file_date)

    @staticmethod
    def get_user_prefix(user_id):
        return user_id[:2].lower().replace('/', '@')

    @staticmethod
    def get_proper_user_id(user_id):
        '''
        The origin user id may have '/' which cannot be output.
        Here change all '/' to '@'
        '''
        return user_id.replace('/', '@')

    @staticmethod
    def get_raw_user_id(user_id):
        return user_id.replace('@', '/')

    def user_based_signal_data(self, file_date):
        '''
        Prepare user based signal data
        :param signal_dir: (str) e.g: "~/hf_signals"
        :param file_date: (str) e.g: "20170607"
        :return:
        '''
        date_pattern = '%Y%m%d'
        data_dir = os.path.join(self.file_dir, 'time_based_signals')
        user_dir = os.path.join(self.file_dir, 'user_based_signals', file_date)
        if os.path.exists(user_dir):
            return

        print(user_dir)
        # TODO: Need the day before and after
        yesterday = (self.get_datetime(file_date, date_pattern) -
                     datetime.timedelta(days=1)).strftime(date_pattern)
        tomorrow = (self.get_datetime(file_date, date_pattern) +
                    datetime.timedelta(days=1)).strftime(date_pattern)
        self.time_based_signal_data(file_date)
        self.time_based_signal_data(yesterday)
        self.time_based_signal_data(tomorrow)

        check_dirs([user_dir])
        files = []
        for d in [x for x in os.listdir(data_dir) if len(x) == 8]:
            tmp_files = [
                os.path.join(data_dir, d, x)
                for x in os.listdir(os.path.join(data_dir, d))
                if file_date in x
            ]
            files.extend(tmp_files)
        files.sort()
        df = None
        t0 = time.time()

        for i, f in enumerate(files):
            print('processing', f)
            tmp = self.get_csv_signals(f)
            if df is None:
                df = tmp
            else:
                df = pd.concat([df, tmp])

            if (i % 4 == 0) or (i == len(files) - 1):
                df['user_prefix'] = [
                    self.get_user_prefix(u) for u in df.user_id
                ]
                unique_prefix = np.unique(df.user_prefix)
                for prefix in unique_prefix:
                    out_f = os.path.join(user_dir, 'user_' + prefix + '.csv')
                    df_tmp = df[df.user_prefix == prefix]
                    df_tmp = df_tmp.drop(['user_prefix'], axis=1)
                    if os.path.exists(out_f):
                        df_tmp.to_csv(out_f,
                                      index=False,
                                      mode='a',
                                      header=False)
                    else:
                        df_tmp.to_csv(out_f, index=False)
                df = None

            if i % 10 == 0:
                print('%d parts is finished. Cost %.2f s.' %
                      (i, time.time() - t0))
                t0 = time.time()
        successed = Json.load(os.path.join(user_dir, 'successed.json'))
        successed['success'].append(file_date)
        Json.output(successed, os.path.join(user_dir, 'successed.json'))
        return

    def user_based_signal_data_async(self, file_dates, process_i=0):
        for file_date in file_dates:
            print('[Process %d] Getting time based signal for date %s' %
                  (process_i, file_date))
            self.user_based_signal_data(file_date)

    def clean_user_signal_data(self, file_date):
        # TODO: clean the signal datas
        data_dir = os.path.join(self.file_dir, 'user_based_signals', file_date)
        self.time_based_signal_data(file_date)

    @staticmethod
    def clean_by_index(obs):
        obs = obs.sort_values('dates')
        obs = obs.reset_index(drop=True)
        return obs

    @staticmethod
    def get_csv_signals(file_name, is_cell=True):
        '''
        Get signals from with csv format.
        :param file_name: (str) e.g: '20170607-0800'
        :param file_dir: signal directory
        :return:
        '''
        df = pd.read_csv(file_name)
        df.dates = df.dates.apply(RawSignal.get_datetime)
        if is_cell:
            df.cell_id = df.cell_id.astype(str)
        return df

    @staticmethod
    def get_datetime(t, pattern=None, no_decimal=True):
        '''Transfer a string time to datetime format.'''
        if no_decimal:
            t = t.split('.')[0]

        if pattern is None:
            if '/' in t:
                pattern = '%Y/%m/%d %H:%M:%S'
            elif '-' in t:
                pattern = '%Y-%m-%d %H:%M:%S'
            else:
                pattern = '%Y%m%d%H%M%S'

        return datetime.datetime.strptime(t, pattern)

    @staticmethod
    def get_cellSheet(signal_dir='',
                      cell_type='baidu',
                      place='hf',
                      filename=None):
        if filename is None:
            cell_dir = os.path.join(signal_dir, '..', 'cellIdSheets')
            filename = os.path.join(
                cell_dir, 'cellIdSheet_' + cell_type + '_' + place + '.txt')
        f = open(filename)
        cell_id, longitude, latitude, radius = [], [], [], []
        for line in f.readlines():
            line_tmp = line.split('\t')
            cell_id.append(line_tmp[0])
            longitude.append(float(line_tmp[1]))
            latitude.append(float(line_tmp[2]))
            radius.append(int(line_tmp[3]))
        f.close()
        return pd.DataFrame({
            'cell_id': cell_id,
            'lon': longitude,
            'lat': latitude,
            'radius': radius
        })

    @staticmethod
    def output_data_js(df, output_f, var_name='data'):
        '''
        output the longitude and latitude data as js file
        :param df: A pandas DataFrame, must contain 'lon' and 'lat' columns.
        :param output_f: the filename to output js file
        :param var_name: Variable name in js file
        :return: No return, just write a js file.
        Js File example:
        var lon = 139
        var lat = 39
        var data = {'points': [[139, 39], [139.1, 39.1]]}
        '''
        lon = df.lon.median()
        lat = df.lat.median()
        max_lon = df.lon.max() * 1.5 - df.lon.min() * 0.5
        min_lon = df.lon.min() * 1.5 - df.lon.max() * 0.5
        max_lat = df.lat.max() * 1.5 - df.lat.min() * 0.5
        min_lat = df.lat.min() * 1.5 - df.lat.max() * 0.5
        N = len(df)
        content = '''var lon = %.5f\nvar lat = %.5f\nvar max_lon = %.5f\nvar min_lon = %.5f''' % (
            lon, lat, max_lon, min_lon) \
                  + '''\nvar max_lat = %.5f\nvar min_lat = %.5f\nvar \n''' % (max_lat, min_lat) \
                  + '''%s = {'points': [''' % var_name
        for i in range(N):
            if 'dates' in df.columns:
                content = '''%s[%.5f, %.5f, '%s']''' % (
                    content, df.iloc[i].lon, df.iloc[i].lat,
                    df.iloc[i].dates.strftime('%Y%m%d%H%M'))
            else:
                content = '''%s[%.5f, %.5f]''' % (content, df.iloc[i].lon,
                                                  df.iloc[i].lat)
            if i != N - 1:
                content = '''%s, ''' % content
            else:
                content = '''%s]}''' % content
        f = open(output_f, 'w')
        f.write(content)
        f.close()

    @staticmethod
    def deal_same_obs(obs):
        '''
        Deal with continuous same observations
        :param obs: DataFrame['dates', 'lon', 'lat']
        :return:
        '''
        # Check DataFrame
        if 'dates' not in obs.columns:
            if 'time' in obs.columns:
                obs['dates'] = obs['time']
            else:
                print(
                    '[Warning] obs DataFrame need "dates" columns. Error in "deal_same_obs function".'
                )
        obs = obs.sort_values(by=['dates'])  # Add 2018-04-17
        obs = obs.reset_index(drop=True)

        # Begin dealing
        inds = [[0]]
        res = obs
        for i in range(1, len(obs)):
            if obs['lon'][i - 1] != obs['lon'][i] or obs['lat'][
                i - 1] != obs['lat'][i]:
                inds.append([i])
            else:
                inds[len(inds) - 1].append(i)
        for ind in inds:
            n = len(ind)
            if n == 1:
                continue
            # mid_time = obs['dates'][ind[0]] + (obs['dates'][ind[n - 1]] - obs['dates'][ind[0]]) / n
            mid_time = obs['dates'][ind[0]] + (obs['dates'][ind[n - 1]] -
                                               obs['dates'][ind[0]]) / 2
            res = res.drop(ind[1:len(ind)])
            res['dates'][ind[0]] = mid_time
        res = res.reset_index(drop=True)
        return res

    @staticmethod
    def norm_pos(pos, alpha=1):
        norm = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
        return pos[0] * alpha / norm, pos[1] * alpha / norm


class AllUsersSignal:
    def __init__(self, signal_dir):
        self.signal_dir = signal_dir
        self.user_dir = os.path.join(signal_dir, 'user_based_signals')
        self.analysis_dir = os.path.join(signal_dir, 'user_analysis')
        self.user_level = [[1, 2], [2, 3], [3, 35], [35, 200], [200, 1000],
                           [1000, 3000], [3000, 10000], [10000, 20000],
                           [20000, 99999999]]

    @staticmethod
    def get_signals(file_date, user_prefix, data_dir):
        '''Given the user prefix and file date get the signals'''
        f = os.path.join(data_dir, file_date, 'user_%s.csv' % user_prefix)
        df = RawSignal.get_csv_signals(f)
        return df

    def user_daily_signal_count_base(self,
                                     file_date,
                                     user_prefix,
                                     unique_cell=False):
        df = self.get_signals(file_date, user_prefix, self.user_dir)
        if unique_cell:
            t = pd.pivot_table(df[['user_id', 'cell_id']],
                               values='cell_id',
                               index='user_id',
                               aggfunc=pd.Series.nunique)
        else:
            t = pd.pivot_table(df[['user_id', 'cell_id']],
                               index='user_id',
                               aggfunc=len)
        counts = t.cell_id.tolist(
        )  # just list(arr) doesn't change the element to int. np.int64 instead.
        # print(counts)
        # self.plot_user_daily_signal_count(counts)
        return counts

    def user_daily_signal_count(self, file_date, plot=False,
                                unique_cell=False):
        '''
        Calculate user daily signal count.
        count = [5, 34, ...]
        :param file_date:
        :param plot: if plot, save the histogram to "analysis_dir/[pic]user_daily_signal_count/*.pdf"
        :param unique_cell: count cell or count unique cell
        :return:
        '''
        check_dir(self.analysis_dir)
        out_name = 'user_daily_signal_count%s.json' % '(unique)' if unique_cell else ''
        out_path = os.path.join(self.analysis_dir, out_name)
        res = {}
        if os.path.exists(out_path):
            res = Json.load(out_path)
            if file_date in res.keys():
                counts = res[file_date]
                print(
                    'User daily signal count in "%s" already existed, skipping ...'
                    % file_date)
                print('Totally %d unique users. Max is %d' %
                      (len(counts), max(counts)))
                print(len([x for x in counts if x > 10000]))
                print(np.mean(counts))
                if plot:
                    self.plot_user_daily_signal_count(counts, file_date)
                return
            else:
                back_up(out_path)
        prefixes = [
            x.split('_')[1][:2]
            for x in os.listdir(os.path.join(self.user_dir, file_date))
            if x.endswith('.csv')
        ]
        counts = []
        N = len(prefixes)
        for i, prefix in enumerate(prefixes):
            counts.extend(
                self.user_daily_signal_count_base(file_date, prefix,
                                                  unique_cell))
            if i % 10 == 0 or i == N - 1:
                print(
                    '[%d/%d] Processing user_%s.csv (%s). avg=%.2f; median=%.2f. %s'
                    % (i, N, prefix, file_date, np.mean(counts),
                       np.median(counts), time.ctime()))
                res[file_date] = counts
                Json.output(
                    res,
                    os.path.join(self.analysis_dir,
                                 '_tmp.'.join(out_name.split('.'))))

        res[file_date] = counts
        Json.output(res, out_path)

        if plot:
            self.plot_user_daily_signal_count(counts, file_date)

    def signal_time_interval_base(self, file_date, user_prefix, max_minute=15):
        df = self.get_signals(file_date, user_prefix, self.user_dir)
        df = df.sort_values(['user_id', 'dates'])
        df = df.reset_index(drop=True)
        # interval = [(df.dates[t+1]-df.dates[t]).total_seconds() for t in range(len(df)-1)
        #             if (df.user_id[t+1] == df.user_id[t]) and (df.cell_id[t+1] != df.cell_id[t])]
        interval = [(df.dates[t + 1] - df.dates[t]).total_seconds()
                    for t in range(len(df) - 1)
                    if df.user_id[t + 1] == df.user_id[t]]
        interval = [t for t in interval if t < max_minute * 60]
        print(np.mean(interval))
        return interval

    def signal_time_interval(self, file_date, plot=False):
        '''
        Calculate user daily signal count.
        count = [5, 34, ...]
        :param file_date:
        :param plot: if plot, save the histogram to "analysis_dir/[pic]user_daily_signal_count/*.pdf"
        :param unique_cell: count cell or count unique cell
        :return:
        Nu5793714
        '''
        check_dir(self.analysis_dir)
        out_name = 'signal_time_interval.json'
        out_path = os.path.join(self.analysis_dir, out_name)
        res = {}
        # if os.path.exists(out_path):
        #     res = Json.load(out_path)
        #     if file_date in res.keys():
        #         counts = res[file_date]
        #         print('User daily signal count in "%s" already existed, skipping ...' % file_date)
        #         print('Totally %d unique users. Max is %d' % (len(counts), max(counts)))
        #         print(len([x for x in counts if x > 10000]))
        #         if plot:
        #             self.plot_user_daily_signal_count(counts, file_date)
        #         return
        #     else:
        #         back_up(out_path)
        prefixes = [
            x.split('_')[1][:2]
            for x in os.listdir(os.path.join(self.user_dir, file_date))
            if x.endswith('.csv')
        ]
        np.random.shuffle(prefixes)
        counts = []
        N = len(prefixes)
        for i, prefix in enumerate(prefixes):
            if i > 100: break
            counts.extend(self.signal_time_interval_base(file_date, prefix))
            if i % 10 == 0 or i == N - 1:
                print(
                    '[%d/%d] Processing user_%s.csv (%s). avg=%.2f; median=%.2f. %s'
                    % (i, N, prefix, file_date, np.mean(counts),
                       np.median(counts), time.ctime()))
                res[file_date] = counts
                Json.output(
                    res,
                    os.path.join(self.analysis_dir,
                                 '_tmp.'.join(out_name.split('.'))))

        res[file_date] = counts
        Json.output(res, out_path)

    def plot_user_daily_signal_count(self,
                                     counts,
                                     file_date=None,
                                     unique_cell=False):
        '''
        Plot user daily signal count histogram.
        :param counts: A counts list
        :param file_date: if file_date is None, then all pictures' title are the same
        :return:
        '''
        kwargs = {
            'title': ('' if file_date is None else '[%s]' % file_date) +
                     'User Daily Signal Count',
            'xlab': 'Count',
            'ylab': 'Number of People'
        }
        out_dir = os.path.join(
            self.analysis_dir, '[pic]user_daily_signal_count%s' %
                               '(unique)' if unique_cell else '')
        check_dir(out_dir)
        AllUsersSignal.plot_count_hist_based(counts, 0, 1000, out_dir,
                                             **kwargs)
        AllUsersSignal.plot_count_hist_based(counts, 0, 200, out_dir, **kwargs)
        AllUsersSignal.plot_count_hist_based(counts, 0, 35, out_dir, **kwargs)
        AllUsersSignal.plot_count_hist_based(counts, 0, 10, out_dir, **kwargs)
        AllUsersSignal.plot_count_hist_based(counts, 0, 15, out_dir, **kwargs)
        AllUsersSignal.plot_count_hist_based(counts, 35, 200, out_dir,
                                             **kwargs)
        AllUsersSignal.plot_count_hist_based(counts,
                                             200,
                                             1000,
                                             out_dir,
                                             bins=200,
                                             **kwargs)
        AllUsersSignal.plot_count_hist_based(counts,
                                             1000,
                                             5000,
                                             out_dir,
                                             bins=200,
                                             **kwargs)
        AllUsersSignal.plot_count_hist_based(counts,
                                             3000,
                                             20000,
                                             out_dir,
                                             bins=200,
                                             **kwargs)

    @staticmethod
    def plot_count_hist_based(counts,
                              start,
                              end,
                              save_dir=None,
                              title='Title',
                              xlab='X',
                              ylab='Y',
                              bins=None):
        kwargs = {
            'title': '_'.join([title, str(start), str(end)]),
            'xlab': xlab,
            'ylab': ylab
        }
        if save_dir is not None:
            kwargs['save_name'] = os.path.join(save_dir,
                                               kwargs['title'] + '.pdf')
        new_counts = [x for x in counts if start < x < end]
        Plot.plot_hist(new_counts,
                       bins=end - start - 1 if bins is None else bins,
                       **kwargs)
        print(
            'Signal counts from %d to %s accounting for %.5f people; %.5f signals'
            % (start, end, len(new_counts) / len(counts),
               sum(new_counts) / sum(counts)))

    def user_daily_signal_count_dict(self, file_date, plot=False):
        '''
        Calculate user daily signal count dictionary.
        res = {'20170623': counts}
        counts = {'1': [user1, user2]}
        :param file_date:
        :param plot: if plot, save the histogram to "analysis_dir/[pic]user_daily_signal_count/*.pdf"
        :return:
        '''
        check_dir(self.analysis_dir)
        out_path = os.path.join(self.analysis_dir,
                                'user_daily_signal_count_dict.json')
        res = {}
        if os.path.exists(out_path):
            res = Json.load(out_path)
            if file_date in res.keys():
                counts = res[file_date]
                print(
                    'User daily signal count dictionary in "%s" already existed, skipping ...'
                    % file_date)
                return
            else:
                back_up(out_path)
        prefixes = [
            x.split('_')[1][:2]
            for x in os.listdir(os.path.join(self.user_dir, file_date))
            if x.endswith('.csv')
        ]
        counts = {}
        N = len(prefixes)
        for i, prefix in enumerate(prefixes):
            if i % 10 == 0 or i == N - 1:
                print('[%d/%d] Processing user_%s.csv (%s). %s' %
                      (i, N, prefix, file_date, time.ctime()))
            df = self.get_signals(file_date, prefix, self.user_dir)
            t = pd.pivot_table(df[['user_id', 'cell_id']],
                               index='user_id',
                               aggfunc=len)
            unique_counts = list(map(str, set(t.cell_id)))
            for c in unique_counts:
                if c in counts.keys():
                    counts[c].extend(list(t[t.cell_id == int(c)].index))
                else:
                    counts[c] = list(t[t.cell_id == int(c)].index)
        res[file_date] = counts
        Json.output(res, out_path)

    def get_diff_level_userids(self, file_date='20170623', seed=223):
        '''
        Get random extracted users from different levels
        Daily signal counts:
        1. 1
        2. 2
        3. 3-34
        4. 35-199
        5. 200-999
        6. 1000-2999
        7. 3000-1w
        8. 1w-2w
        9. all > 2w
        :return:
            Output chosen different user ids
            {user_level: [selected user_id]}
            user_level example: 3_35 means signal counts ranging from 3 to 34
        '''
        print('Getting different level users dictionary')
        data_path = os.path.join(self.analysis_dir,
                                 'user_daily_signal_count_dict.json')
        output_path = os.path.join(self.analysis_dir,
                                   'different_level_users_%s.json' % file_date)
        if os.path.exists(output_path):
            print('Already exists "%s". Skipping ...' % output_path)
            return
        users = Json.load(data_path)[file_date]
        user_level = self.user_level
        chosen = {'_'.join(map(str, x)): [] for x in user_level}
        choose_num = 50
        np.random.seed(seed)
        for level in user_level[:-1]:
            c = []
            while len(c) < choose_num:
                cnt = np.random.randint(level[0], level[1])
                if str(cnt) in users.keys():
                    c.append(np.random.choice(users[str(cnt)]))
            chosen['_'.join(map(str, level))] = c
        # record all the users who have more than 2w signals
        c = []
        for cnt in [x for x in users.keys() if int(x) > 20000]:
            c.extend(users[cnt])
        chosen['_'.join(map(str, user_level[-1]))] = c
        Json.output(chosen, output_path)

    def get_different_user_level_user_signals(self, user_date='20170623'):
        '''
        Get random users daily signals. The users are chosen from 'get_diff_level_userids'.
        output in the "os.path.join(self.analysis_dir, 'different_level_users')" directory
        :param user_date:
        :return:
        '''
        print('getting different level users signals')
        self.get_diff_level_userids(user_date)
        user_path = os.path.join(self.analysis_dir,
                                 'different_level_users_%s.json' % user_date)
        work_dir = os.path.join(self.analysis_dir, 'different_level_users')
        detail_dir = os.path.join(work_dir, 'details')
        check_dirs([work_dir, detail_dir])
        users = Json.load(user_path)
        all_users = []
        for level in users.keys():
            all_users.extend(users[level])
        user_prefix = list(
            set([RawSignal.get_user_prefix(u) for u in all_users]))
        available_dates = os.listdir(self.user_dir)

        # Output the random selected users' daily signals
        #
        for d in available_dates:
            output_path = os.path.join(
                work_dir, '[%s]user_signals_based_on_%s.csv' % (d, user_date))
            if os.path.exists(output_path):
                print('"%s" already exists, skipping ...' % output_path)
                continue
            print('Getting %s users signals' % d)
            df = None
            for i, prefix in enumerate(user_prefix):
                if i % 5 == 0:
                    print('Process %s user signal: %d/%d prefix=%s (%s)' %
                          (d, i, len(user_prefix), prefix, time.ctime()))
                tmp = RawSignal.get_csv_signals(
                    os.path.join(self.user_dir, d, 'user_%s.csv' % prefix))
                tmp_user = [
                    u for u in all_users
                    if RawSignal.get_user_prefix(u) == prefix
                ]
                tmp = tmp[tmp['user_id'].isin(tmp_user)]
                if df is None:
                    df = tmp
                else:
                    df = pd.concat([df, tmp])
            df.to_csv(output_path, index=False)

    def show_different_user_level_user_signals(self, user_date='20170623'):
        # self.get_different_user_level_user_signals(user_date)
        work_dir = os.path.join(self.analysis_dir, 'different_level_users')
        # check_dirs([os.path.join(work_dir, 'level'+'_'.join(map(str, x))) for x in self.user_level])
        detail_dir = os.path.join(work_dir, 'detail_user_signals')
        json_dir = os.path.join(work_dir, 'json_user_signals')
        html_dir = os.path.join(work_dir, 'html_user_signals')
        check_dirs([detail_dir, json_dir, html_dir])
        files = sorted([x for x in os.listdir(work_dir) if x.endswith('csv')])
        print(files)
        cells = RawSignal.get_cellSheet(self.signal_dir)
        df = None
        for f in files:
            tmp = RawSignal.get_csv_signals(os.path.join(work_dir, f))
            if df is None:
                df = tmp
            else:
                df = pd.concat([df, tmp])

        df = pd.merge(df, cells)

        users = sorted(list(set(df.user_id)))
        print('Totally %d users to generate' % len(users))
        for u_i, user in enumerate(users):
            print('Processing %d/%d users' % (u_i, len(users)))
            tmp = df[df.user_id == user]
            tmp = tmp.sort_values(by='dates')
            n = int(len(tmp) / len(files))
            # tmp.to_csv(os.path.join(detail_dir, '[%.6d]user%d.csv' % (n, u_i)), index=False)
            tmp = tmp[['dates', 'lon', 'lat']]
            # RawSignal.output_data_js(tmp, os.path.join(json_dir, '[%.6d]user%d.json' % (n, u_i)))
            MapPlot.plot_traj([tmp.lon.tolist()], [tmp.lat.tolist()],
                              zoom=13,
                              save_path=os.path.join(
                                  html_dir, '[%.6d]user%d.html' % (n, u_i)))


class Json:
    @staticmethod
    def load(file_path):
        '''
        Load json file
        :param file_path: file path
        :return: A dictionary
        '''
        with open(file_path) as f:
            out = json.load(f)
        return out

    @staticmethod
    def output(out, file_path):
        '''
        Output dictionary to a file path
        :param out: A dictionary
        :param file_path:
        :return: None
        '''
        with open(file_path, 'w') as f:
            json.dump(out, f)
        return


class UserSignal:
    def __init__(self, user_id, signal_dir):
        self.user_id = user_id
        self.prefix = user_id[:2].lower().replace('/', '@')
        self.signal_dir = signal_dir
        self.user_dir = os.path.join(signal_dir, 'user_based_signals')

    def get_signal(self, file_date):
        '''Given the user id and file date get this user's signal'''
        df = AllUsersSignal.get_signals(file_date, self.prefix, self.user_dir)
        df = df[df.user_id == self.user_id]
        df = df.sort_values(by=['dates'])
        print(df.head())

    @staticmethod
    def load_signal_user(user_based_signal_dir, user_id):
        prefix = RawSignal.get_user_prefix(user_id)
        signal = RawSignal.get_csv_signals(
            os.path.join(user_based_signal_dir, 'user_%s.csv' % prefix))
        signal = signal[signal.user_id == user_id]
        signal = signal.sort_values('dates')
        return signal


class Plot:
    @staticmethod
    def plot_3d_dist(hist, xedges, yedges, save_name=None):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.backends.backend_pdf import PdfPages
        if save_name is not None:
            pdf = PdfPages(save_name)
            fig = plt.figure()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
        xpos = xpos.flatten('F')
        ypos = ypos.flatten('F')
        zpos = np.zeros_like(xpos)

        # Construct arrays with the dimensions for the 16 bars.
        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()
        dz = hist.flatten()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

        if save_name is not None:
            pdf.savefig(fig)
        else:
            plt.show()

    @staticmethod
    def plot_hist(lst,
                  xlim,
                  ylim,
                  title='Title',
                  xlab='X',
                  ylab='Y',
                  bins=100,
                  save_name=None):
        import seaborn as sns
        from matplotlib.backends.backend_pdf import PdfPages
        if save_name is not None:
            pdf = PdfPages(save_name)
            fig = plt.figure()
        sns.set_palette('deep', desat=.6)
        sns.set_context(rc={'figure.figsize': (24, 15)})
        plt.subplots_adjust(top=0.9, right=0.95, left=0.15, bottom=0.1)
        plt.hist(lst, bins=bins)
        plt.grid(True, linestyle="--", color="black", linewidth=".5", alpha=.4)
        plt.title(title)
        plt.xlim((xlim[0], xlim[1]))
        plt.ylim((ylim[0], ylim[1]))
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        if save_name is not None:
            pdf.savefig(fig)
        else:
            plt.show()

    @staticmethod
    def plot_lines(x, y, title, xlab, ylab, save_name=None):
        plt.figure(figsize=(15, 3))
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.show()
        if save_name is not None:
            pdf.savefig(fig)
        else:
            plt.show()

    @staticmethod
    def plot_schedule(x,
                      y,
                      s,
                      c,
                      xlab,
                      ylab,
                      gap,
                      title=None,
                      save_name=None,
                      labels=None,
                      legend=None,
                      text_size=20):
        font5 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 30}
        font4 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 26}
        font1 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 18}
        font3 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 23}
        font6 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 28}
        font7 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 34}
        font2 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 20}
        font_size = 20
        plt.style.use('ggplot')
        fig1 = plt.figure(figsize=(15, 3))
        for i in range(len(x)):
            plt.scatter(x[i], y[i], c=c[i], s=s)
        if title is not None:
            plt.title(title, fontsize=text_size)
        plt.xlabel(xlab, font1)
        plt.ylabel(ylab, font1)
        if labels is not None:
            plt.legend(labels=labels, loc='upper left', fontsize=text_size + 1)
        hour_num = [x / 2 for x in range(int(gap[0]) * 2, int(gap[1] * 2) + 1, 1)]
        hour_str = [
            str(int(x)) + ':00' if x % 1 == 0 else str(int(x)) + ':30'
            for x in hour_num
        ]

        plt.xlim((gap[0], gap[1]))
        plt.xticks(hour_num, hour_str)
        plt.xticks(fontproperties='STIXGeneral', size=font_size - 10)
        plt.yticks(fontproperties='STIXGeneral', size=font_size - 1)
        plt.show()

    @staticmethod
    def plot_one_schedule(x,
                          y,
                          s,
                          c,
                          xlab,
                          ylab,
                          gap,
                          title=None,
                          save_name=None,
                          labels=None,
                          legend=None,
                          text_size=20):
        font5 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 30}
        font4 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 26}
        font1 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 18}
        font3 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 23}
        font6 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 28}
        font7 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 34}
        font2 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 20}
        font_size = 20
        plt.style.use('ggplot')
        fig1 = plt.figure(figsize=(15, 3))

        plt.scatter(x, y, c=c, s=s)

        if title is not None:
            plt.title(title, fontsize=text_size)
        plt.xlabel(xlab, font1)
        plt.ylabel(ylab, font1)
        if labels is not None:
            plt.legend(labels=labels, loc='upper left', fontsize=text_size + 1)
        hour_num = [x / 2 for x in range(gap[0] * 2, int(gap[1] * 2) + 1, 1)]
        hour_str = [
            str(int(x)) + ':00' if x % 1 == 0 else str(int(x)) + ':30'
            for x in hour_num
        ]

        plt.xlim((gap[0], gap[1]))
        plt.xticks(hour_num, hour_str)
        plt.xticks(fontproperties='STIXGeneral', size=font_size - 10)
        plt.yticks(fontproperties='STIXGeneral', size=font_size - 1)
        plt.show()

    @staticmethod
    def plot_bidensity(dat, x='x', y='y'):
        # dat is a [[[lon, lat]], [[lon, lat]]]
        import seaborn as sns
        for d in dat:
            dist_2d = [x for x in d if abs(x[0]) < 800 and abs(x[1]) < 800]
            dist_2d = [[dist_2d[i][j] for i in range(len(dist_2d))]
                       for j in range(2)]  # [[lon], [lat]]
            sns.jointplot(x=x,
                          y=y,
                          data=pd.DataFrame({
                              x: dist_2d[0],
                              y: dist_2d[1]
                          }))
        plt.show()

    @staticmethod
    def plot_density(dat,
                     x='x',
                     y='y',
                     title='',
                     labels=None,
                     label_size=24,
                     text_size=20,
                     colors=None,
                     hist=True):
        # dat is a [[dist]]
        if colors is None:
            colors = [None] * len(dat)
        if labels is None:
            labels = [None] * len(dat)
        import seaborn as sns
        for i, d in enumerate(dat):
            sns.distplot(np.array(d),
                         color=colors[i],
                         hist=hist,
                         label=labels[i])
        plt.subplots_adjust(top=0.97, right=0.98, left=0.13, bottom=0.15)
        plt.title(title, fontweight='bold', fontsize=14)
        plt.xlabel(x, fontsize=label_size, fontweight='bold')
        plt.ylabel(y, fontsize=label_size, fontweight='bold')
        plt.yticks(fontsize=text_size, fontweight='bold')
        plt.xticks(fontsize=text_size, fontweight='bold')
        if labels is not None:
            plt.legend(labels=labels, loc='best', fontsize=text_size)
        plt.show()

    @staticmethod
    def boxplot(dat,
                xlabel,
                ylabel,
                labels,
                title='',
                label_size=24,
                text_size=20):
        # dat is a [[x]]
        fig, ax = plt.subplots()
        pos = np.array(range(len(dat))) + 1
        bp = ax.boxplot(dat, sym='k+', positions=pos, notch=1, bootstrap=5000)
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=label_size)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=label_size)
        ax.set_xticklabels(labels,
                           rotation=0,
                           fontsize=text_size,
                           fontweight='bold')
        plt.subplots_adjust(top=0.97, right=0.98, left=0.15, bottom=0.15)
        plt.yticks(fontsize=text_size, fontweight='bold')
        plt.title(title, fontweight='bold', fontsize=label_size)
        plt.ylim(0, 700)
        plt.setp(bp['whiskers'], color='k', linestyle='-')
        plt.setp(bp['fliers'], markersize=3.0)
        plt.show()


class Line:
    @staticmethod
    def dist_pos2line(line: LineString, pos):
        cal_distance(pos, Line.pos_project_line(line, pos).coords[0])

    @staticmethod
    def pos_project_line(line: LineString, pos):
        return line.interpolate(line.project(Point(
            pos[0], pos[1]))).coords[0]  # a position [lon, lat]


class List:
    def __init__(self, lst):
        self.lst = lst

    def min_ind(self):
        lst = self.lst
        if len(lst) == 0:
            return -1
        ind, tmp = 0, lst[0]
        for i in range(1, len(lst)):
            if lst[i] < tmp:
                tmp = lst[i]
                ind = i
        return ind

    def max_ind(self):
        lst = [-x for x in self.lst]
        return List(lst).min_ind()


class NewDict(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(NewDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(NewDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(NewDict, self).__delitem__(key)
        del self.__dict__[key]


class MapPlot:
    def __init__(self):
        pass

    @staticmethod
    def output_data_js(df, filename, user_id=-1, var_name='data'):
        '''
        output the longitude and latitude data as js file
        :param df: A pandas DataFrame, must contain 'lon' and 'lat' columns.
        :param filename: the filename to output js file
        :param var_name: Variable name in js file
        :return: No return, just write a js file.
        Js File example:
        var lon = 139
        var lat = 39
        var data = {'points': [[139, 39], [139.1, 39.1]]}
        '''
        lon = df.lon.median()
        lat = df.lat.median()
        max_lon = df.lon.max() * 1.5 - df.lon.min() * 0.5
        min_lon = df.lon.min() * 1.5 - df.lon.max() * 0.5
        max_lat = df.lat.max() * 1.5 - df.lat.min() * 0.5
        min_lat = df.lat.min() * 1.5 - df.lat.max() * 0.5
        N = len(df)
        content = '''var lon = %.5f\nvar lat = %.5f\nvar max_lon = %.5f\nvar min_lon = %.5f''' % (
            lon, lat, max_lon, min_lon) \
                  + '''\nvar max_lat = %.5f\nvar min_lat = %.5f\nvar user_id = '%s'\n''' % (max_lat, min_lat, user_id) \
                  + '''%s = {'points': [''' % var_name
        for i in range(N):
            if 'dates' in df.columns:
                content = '''%s[%.5f, %.5f, '%s']''' % (
                    content, df.iloc[i].lon, df.iloc[i].lat,
                    str(df.iloc[i].dates))
            else:
                content = '''%s[%.5f, %.5f]''' % (content, df.iloc[i].lon,
                                                  df.iloc[i].lat)
            if i != N - 1:
                content = '''%s, ''' % content
            else:
                content = '''%s]}''' % content
        f = open(filename, 'w')
        f.write(content)
        f.close()

    @staticmethod
    def _bd2folium(lons, lats):
        cor = Coordinate()
        points = [cor.bd2gcj(lons[i], lats[i]) for i in range(len(lons))]
        lons = [x[0] - 0.0055 for x in points]
        lats = [x[1] + 0.002 for x in points]
        return lons, lats

    @staticmethod
    def round_lst(lst, rnd):
        return [round(x, rnd) for x in lst]

    @staticmethod
    def plot_points(lons,
                    lats,
                    save_path,
                    zoom=12,
                    transform=False,
                    color='red',
                    radius=3):
        '''
        Plot cell positions
        :param lons: A list of longitude
        :param lats: A list of latitude
        :param save_path: path to save graph
        :param zoom: The zoom of the map
        :param transform: If the lon is google lon, no need to transform. If the lon is baidu lon, need transform
        :return:
        '''
        nlons = [
            float(lons[i]) for i in range(len(lons))
            if lons[i] != '' and lats[i] != ''
        ]
        nlats = [
            float(lats[i]) for i in range(len(lons))
            if lons[i] != '' and lats[i] != ''
        ]
        lons, lats = nlons, nlats
        m_lon = np.mean(lons)
        m_lat = np.mean(lats)
        my_map = folium.Map(location=[m_lat, m_lon], zoom_start=zoom)
        if transform:
            lons, lats = MapPlot._bd2folium(lons, lats)
        for i in range(len(lons)):
            # folium.Marker(location=[lats[i], lons[i]]).add_to(my_map)
            folium.CircleMarker(location=[lats[i], lons[i]],
                                radius=radius,
                                fill=True,
                                color=color,
                                fillOpacity=1).add_to(my_map)
        my_map.fit_bounds(my_map.get_bounds())
        my_map.save(save_path)

    @staticmethod
    def pop_mark(lon, lat, mp, info='', color='blue'):
        colors = MARK_COLOR
        color = color if type(color) == str else colors[color % len(colors)]
        if info == '':
            test = folium.Html('lon:{}</br> lat:{}</br> '.format(lon, lat),
                               script=True)
        else:
            test = folium.Html('info:{}</br> lon:{}</br> lat:{}</br> '.format(
                info, lon, lat),
                script=True)
        popup = folium.Popup(test, max_width=1000)
        folium.Marker([lat, lon], popup=popup,
                      icon=folium.Icon(color=color)).add_to(mp)
        return mp

    @staticmethod
    def plot_signal_user(user_id, signal_dir, save_path):
        signal = UserSignal.load_signal_user(signal_dir, user_id)
        cells = RawSignal.get_cellSheet(os.path.join(signal_dir, '..', '..'))
        signal = pd.merge(signal, cells)
        signal = signal.sort_values('dates')
        signal = signal.reset_index(drop=True)
        print(signal)
        MapPlot.plot_traj_by_obs(signal,
                                 save_path=save_path,
                                 marker=True,
                                 dates=signal.dates.tolist())

    @staticmethod
    def plot_traj(lons, lats, zoom=10, colors=None, save_path='a.html', transforms=None, rnd=8, dates=None,
                  marker=False):
        '''Input is baidu, folium is transform of gcj'''
        from folium import plugins

        if len(lons) > 0 and type(lons[0]) == list:
            n_traj = len(lons)
            lon_m = round(np.sum(np.sum(lons)) / np.sum([len(x) for x in lons]), rnd)
            lat_m = round(np.sum(np.sum(lats)) / np.sum([len(x) for x in lats]), rnd)
            my_map = folium.Map(location=[lat_m, lon_m], zoom_start=zoom)
            transforms = [True] * n_traj if transforms is None else transforms if type(transforms) == list else [
                                                                                                                    transforms] * len(
                lons)
            colors = ['black'] * n_traj if colors is None else colors if type(colors) == list else [colors] * n_traj
            colors = (colors * (n_traj // len(colors) + 1))[:n_traj] if len(colors) != n_traj else colors
            # if marker: colors = [MARK_COLOR[i % len(MARK_COLOR)] for i in range(n_traj)]
            for traj in range(len(lons)):
                if transforms[traj]:
                    lon, lat = MapPlot._bd2folium(lons[traj], lats[traj])
                else:
                    lon, lat = lons[traj], lats[traj]
                lon = MapPlot.round_lst(lon, rnd)
                lat = MapPlot.round_lst(lat, rnd)
                folium.PolyLine([(lat[i], lon[i]) for i in range(len(lons[traj]))],
                                color='black' if colors is None else colors[traj]).add_to(my_map)
                # my_map = MapPlot.pop_mark(lon[0], lat[0], my_map)
            if marker:
                poses = {}
                for traj in range(len(lons)):
                    if transforms[traj]:
                        lon, lat = MapPlot._bd2folium(lons[traj], lats[traj])
                    else:
                        lon, lat = lons[traj], lats[traj]
                    for i in range(len(lons[traj])):
                        pos = (lon[i], lat[i])
                        tmp = [traj, i] if dates is None else [traj, i, dates[traj][i]]
                        if pos in poses:
                            poses[pos].append(tmp)
                        else:
                            poses[pos] = [tmp]
                for pk, pv in poses.items():
                    if dates is not None:
                        pvs = ['T%d; i%d; %s' % (v[0], v[1], str(v[2])) for v in pv]
                        my_map = MapPlot.pop_mark(pk[0], pk[1], my_map, 'Point: </br>%s' % '</br>'.join(pvs),
                                                  color=pv[0][0])
                    else:
                        pvs = ['T%d; i%d' % (v[0], v[1]) for v in pv]
                        my_map = MapPlot.pop_mark(pk[0], pk[1], my_map, "Point: </br>%s" % '</br>'.join(pvs),
                                                  color=pv[0][0])
            my_map.fit_bounds(my_map.get_bounds())
            my_map.save(save_path)
        elif len(lons) > 0:
            assert type(colors) != list
            assert type(transforms) != list
            lon_m, lat_m = np.mean(lons), np.mean(lats)
            my_map = folium.Map(location=[lat_m, lon_m], zoom_start=zoom)
            transforms = True if transforms is None else transforms
            if transforms:
                lons, lats = MapPlot._bd2folium(lons, lats)
            lons = MapPlot.round_lst(lons, rnd)
            lats = MapPlot.round_lst(lats, rnd)
            folium.PolyLine([(lats[i], lons[i]) for i in range(len(lons))],
                            color='black' if colors is None else colors).add_to(my_map)
            if marker:
                poses = {}
                for i in range(len(lons)):
                    pos = (lons[i], lats[i])
                    tmp = [i] if dates is None else [i, dates[i]]
                    if pos in poses:
                        poses[pos].append(tmp)
                    else:
                        poses[pos] = [tmp]
                for pk, pv in poses.items():
                    if dates is not None:
                        pvs = ['i%d; %s' % (v[0], str(v[1])) for v in pv]
                        my_map = MapPlot.pop_mark(pk[0], pk[1], my_map, 'Point: </br>%s' % '</br>'.join(pvs),
                                                  color=pv[0][0])
                    else:
                        pvs = ['i%d' % v[0] for v in pv]
                        my_map = MapPlot.pop_mark(pk[0], pk[1], my_map, "Point: </br>%s" % '</br>'.join(pvs),
                                                  color=pv[0][0])
            my_map.fit_bounds(my_map.get_bounds())
            my_map.save(save_path)

    @staticmethod
    def plot_road(nodes,
                  ways,
                  zoom=10,
                  colors=None,
                  save_path='a.html',
                  transforms=None,
                  rnd=8,
                  points=False,
                  cross_point=False):
        '''Input is baidu, folium is transform of gcj
        nodes: {node_id: [lon, lat]}
        ways: {link_id: [nodes]}
        '''
        from folium import plugins
        lons = [[float(nodes[x][0]) for x in v] for k, v in ways.items()]
        lats = [[float(nodes[x][1]) for x in v] for k, v in ways.items()]
        n_traj = len(lons)
        lon_m = round(
            np.sum(np.sum(lons)) / np.sum([len(x) for x in lons]), rnd)
        lat_m = round(
            np.sum(np.sum(lats)) / np.sum([len(x) for x in lats]), rnd)
        my_map = folium.Map(location=[lat_m, lon_m], zoom_start=zoom)
        transforms = [True] * n_traj if transforms is None else transforms if type(transforms) == list \
            else [transforms] * len(lons)
        colors = ['red'] * n_traj if colors is None else colors if type(
            colors) == list else [colors] * n_traj
        colors = (colors * (n_traj // len(colors) + 1)
                  )[:n_traj] if len(colors) != n_traj else colors
        for traj in range(len(lons)):
            if transforms[traj]:
                lon, lat = MapPlot._bd2folium(lons[traj], lats[traj])
            else:
                lon, lat = lons[traj], lats[traj]
            lon = MapPlot.round_lst(lon, rnd)
            lat = MapPlot.round_lst(lat, rnd)
            folium.PolyLine([(lat[i], lon[i]) for i in range(len(lons[traj]))],
                            color='black'
                            if colors is None else colors[traj]).add_to(my_map)
            if points:
                for i in range(len(lon)):
                    # my_map = MapPlot.pop_mark(lon[0] * 0.9 + lon[1] * 0.1, lat[0] * 0.9 + lat[1] * 0.1, my_map)
                    # my_map = MapPlot.pop_mark(lon[i] * 0.9 + lon[i+1] * 0.1, lat[i] * 0.9 + lat[i+1] * 0.1, my_map)
                    my_map = MapPlot.pop_mark(lon[i], lat[i], my_map)

        if cross_point:
            cross_nodes = {k: 0 for k in nodes}
            for v in ways.values():
                for vv in v:
                    cross_nodes[vv] += 1
            cross_nodes = {
                k: list(nodes[k]) + [v]
                for k, v in cross_nodes.items() if v > 1
            }
            clons = [v[0] for v in cross_nodes.values()]
            clats = [v[1] for v in cross_nodes.values()]
            nums = [v[2] for v in cross_nodes.values()]
            color_fn = lambda x: 'brown' if x == 2 else 'pink' if x == 3 else 'orange' if x == 4 else 'green'
            for i in range(len(clons)):
                folium.CircleMarker(location=[clats[i], clons[i]],
                                    radius=3,
                                    fill=True,
                                    color=color_fn(nums[i]),
                                    fillOpacity=1).add_to(my_map)
        my_map.fit_bounds(my_map.get_bounds())
        my_map.save(save_path)

    @staticmethod
    def plot_traj_by_obs(obs,
                         zoom=10,
                         colors=None,
                         save_path='a.html',
                         transforms=None,
                         rnd=8,
                         marker=False,
                         dates=None):
        '''obs is pd.DataFrame with 'lon' and 'lat' attribute.'''
        lons = obs.lon.tolist()
        lats = obs.lat.tolist()
        MapPlot.plot_traj(lons,
                          lats,
                          zoom,
                          colors,
                          save_path,
                          transforms,
                          rnd,
                          marker=marker,
                          dates=dates)

    @staticmethod
    def plot_traj_by_obs_per_day(obs,
                                 save_path,
                                 colors=None,
                                 transforms=None,
                                 rnd=8,
                                 marker=True):
        obs['day'] = [x.strftime('%Y%m%d') for x in obs.dates.tolist()]
        out_dir = os.path.join(save_path[:-5])
        check_dirs(out_dir)

        lons, lats, dates = [], [], []
        for da in np.unique(obs.day):
            tmp = obs[obs.day == da]
            tmp = RawSignal.clean_by_index(tmp)
            lons.append(tmp.lon.tolist())
            lats.append(tmp.lat.tolist())
            dates.append(tmp.dates.tolist())
            MapPlot.plot_traj_by_obs(
                tmp,
                save_path=os.path.join(out_dir, da + '.html'),
                transforms=transforms,
                rnd=rnd,
                marker=marker,
                dates=tmp.dates.tolist() if marker else None)
        MapPlot.plot_traj(lons,
                          lats,
                          10,
                          colors,
                          save_path[:-5] + '_%s.html',
                          transforms,
                          rnd,
                          marker=marker,
                          dates=dates if marker else None)


class RoadTest:
    def __init__(self, test_dir, cell_path):
        self.test_dir = test_dir
        self.cells = RawSignal.get_cellSheet(filename=cell_path)

    def get_wuhu_ground_truth(self, trail=0, direction=0):
        csv_dir = os.path.join('..', 'BigEyeOut', 'wuhu_test', 'ground_truth',
                               'json')
        d = Json.load(
            os.path.join(csv_dir, 'ground_truth_trail%d.json' % trail))
        if direction == 0:
            df = pd.DataFrame({'lon': d['lon'], 'lat': d['lat']})
        else:
            df = pd.DataFrame({'lon': d['lon'][::-1], 'lat': d['lat'][::-1]})
        return df

    def all_road_config(self):
        roads = []
        roads_d = {}
        for trail in range(1, 9):
            for direction in [-1, 1]:
                for n in range(1, 11):
                    roads.append({
                        'trail': trail,
                        'direction': direction,
                        'n': n
                    })
        for k in roads[0].keys():
            roads_d[k] = [roads[i][k] for i in range(len(roads))]
        return roads, roads_d

    def get_road_tests(self, trail=0, direction=0, n=None, simple=True):
        '''
        get road test data of a trail with direction.
        :param trail: <Int> The i_th trail. If 0, don't consider trail and n.
        :param direction: <-1/1> -1 means reverse side; 1 means front; 0 means don't consider direction.
        :param n: The n_th test of this trail. If None, return all the tests of the trail.
        :return: A pandas DataFrame
        '''

        headers = [
            'dates', 'nothing', 'cell_id', 'user_id', 'service_type', 'web',
            'glon', 'glat'
        ]

        n = '%d.csv' % n if n is not None else ''
        trail = u'线路' + str(trail) if trail != 0 else ''
        direction = u'正向' if direction == 1 else u'逆向' if direction == -1 else ''
        direction = direction + n
        files_ind = [
            trail in file and direction in file and file.endswith('.csv')
            for file in os.listdir(self.test_dir)
        ]
        files = np.array(os.listdir(self.test_dir))[files_ind]
        print(files)
        # df = pd.read_csv(mypath + files[0], names = headers)
        for file in files:
            tmp = pd.read_csv(os.path.join(self.test_dir, file), names=headers)
            tmp = self.remove_null(tmp, ['glon', 'glat'])
            if 'data' in locals():
                data = pd.concat([data, tmp])
            else:
                data = tmp
        data = data.drop(['nothing'], axis=1)
        data = data.dropna(axis=0, how='any')
        data = data.reset_index(drop=True)
        data.dates = data.dates.apply(self.get_time)
        data.glon = data.glon.apply(float) + 0.012
        data.glat = data.glat.apply(float) + 0.0045
        data.cell_id = data.cell_id.apply(str)
        data = pd.merge(data, self.cells)
        data = data.sort_values('dates')
        data = data.reset_index(drop=True)
        if simple:
            data = data[['dates', 'glon', 'glat', 'lon', 'lat']]
        return data

    def get_all_wuhu(self, simple=False):
        df = None
        for trail in range(1, 9):
            for d in [-1, 1]:
                for n in range(1, 11):
                    tmp = self.get_road_tests(trail, d, n, simple=simple)
                    tmp['user_id'] = '_'.join(list(map(str, [trail, d, n])))
                    if df is None:
                        df = tmp
                    else:
                        df = pd.concat([df, tmp])
        return df

    @staticmethod
    def remove_null(df: pd.DataFrame, columns):
        '''
        Remove samples if there is 'null' in given columns
        :param df: A DataFrame
        :param columns: A string list. Columns to check whether 'null' in them.
        :return:
        '''
        for col in columns:
            if df.dtypes[col] == 'O':
                df = df.iloc[[not x for x in df[col].isin(['null'])]]
        return df.reset_index(drop=True)

    @staticmethod
    def get_time(t):
        '''
        Transform yyyymmddhhmmss [Int] to a datetime type
        :param t: Time (yyyymmddhhmmss)
        :return: datetime
        e.x: df = [time, lon, lat]
        1. df.time = df.time.apply(get_time)
        2. df.time = [get_time(x) for x in df.time]
        '''
        yy = int(t // 1e10)
        mm = int(t % 1e10 // 1e8)
        dd = int(t % 1e8 // 1e6)
        hh = int(t % 1e6 // 1e4)
        min = int(t % 1e4 // 1e2)
        s = int(t % 1e2)
        return datetime.datetime(yy, mm, dd, hh, min, s)


class Coordinate:
    def __init__(self):
        self.a = 6378245.0
        self.ee = 0.00669342162296594323
        self.x_pi = 3.14159265358979324 * 3000.0 / 180.0
        self.cor_types = ['gcj', 'baidu', 'wgs']

    @staticmethod
    def transformLon(lat, lon):
        # 转换纬度
        ret = 300.0 + lat + 2.0 * lon + 0.1 * lat * lat + 0.1 * lat * lon + 0.1 * math.sqrt(
            abs(lat))
        ret += (20.0 * math.sin(6.0 * lat * math.pi) +
                20.0 * math.sin(2.0 * lat * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * math.pi) +
                40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lat / 12.0 * math.pi) +
                300.0 * math.sin(lat / 30.0 * math.pi)) * 2.0 / 3.0
        return ret

    @staticmethod
    def transformLat(lat, lon):
        # 转换经度
        ret = -100.0 + 2.0 * lat + 3.0 * lon + 0.2 * lon * lon + 0.1 * lat * lon + 0.2 * math.sqrt(
            abs(lat))
        ret += (20.0 * math.sin(6.0 * lat * math.pi) +
                20.0 * math.sin(2.0 * lat * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lon * math.pi) +
                40.0 * math.sin(lon / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lon / 12.0 * math.pi) +
                320 * math.sin(lon * math.pi / 30.0)) * 2.0 / 3.0
        return ret

    def gcj2bd(self, lon, lat, dlon=0.0065, dlat=0.006):
        '''From GCJ02 -> Baidu'''
        x = lon
        y = lat
        z = math.sqrt(x * x + y * y) + 0.00002 * math.sin(y * self.x_pi)
        theta = math.atan2(y, x) + 0.000003 * math.cos(x * self.x_pi)
        bd_lon = z * math.cos(theta) + dlon
        bd_lat = z * math.sin(theta) + dlat
        bdpoint = [bd_lon, bd_lat]
        return bdpoint

    def gcj2bd_raw(self, lon, lat):
        return self.gcj2bd(lon, lat, 0, 0)

    def wgs2gcj(self, lon, lat):
        dLat = self.transformLat(lon - 105.0, lat - 35.0)
        dLon = self.transformLon(lon - 105.0, lat - 35.0)
        radLat = lat / 180.0 * math.pi
        magic = math.sin(radLat)
        magic = 1 - self.ee * magic * magic
        sqrtMagic = math.sqrt(magic)
        dLat = (dLat * 180.0) / ((self.a * (1 - self.ee)) /
                                 (magic * sqrtMagic) * math.pi)
        dLon = (dLon * 180.0) / (self.a / sqrtMagic * math.cos(radLat) *
                                 math.pi)
        mgLat = lat + dLat
        mgLon = lon + dLon
        loc = [mgLat, mgLon]
        return loc

    def gcj2wgs(self, lng, lat):
        """
        GCJ02(火星坐标系)转GPS84
        :param lng:火星坐标系的经度
        :param lat:火星坐标系纬度
        :return:
        """
        dlat = self.transformLat(lng - 105.0, lat - 35.0)
        dlng = self.transformLon(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * self.x_pi
        magic = math.sin(radlat)
        magic = 1 - self.ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((self.a * (1 - self.ee)) /
                                 (magic * sqrtmagic) * self.x_pi)
        dlng = (dlng * 180.0) / (self.a / sqrtmagic * math.cos(radlat) *
                                 self.x_pi)
        mglat = lat + dlat
        mglng = lng + dlng
        return [lng * 2 - mglng, lat * 2 - mglat]

    def bd2gcj(self, bd_lon, bd_lat):
        """
        百度坐标系(BD-09)转火星坐标系(GCJ-02)
        百度——>谷歌、高德
        :param bd_lat:百度坐标纬度
        :param bd_lon:百度坐标经度
        :return:转换后的坐标列表形式
        """
        x = bd_lon - 0.0065
        y = bd_lat - 0.006
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * self.x_pi)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * self.x_pi)
        gg_lng = z * math.cos(theta)
        gg_lat = z * math.sin(theta)
        return [gg_lng, gg_lat]

    def bd2wgs(self, lon, lat):
        gcj = self.bd2gcj(lon, lat)
        return self.gcj2wgs(gcj[0], gcj[1])

    def eval_bd_wgs(self, lon, lat):
        nlon, nlat = self.gcj2bd(lon, lat)
        nlon, nlat = self.bd2gcj(nlon, nlat)
        print(lon, lat, nlon, nlat)
        assert abs(lon - nlon) < 0.00001


class Metrics:
    @staticmethod
    def save_all_metric(pred, label, out=None, prnt=False):
        '''
        :param out: A dictionary with pcs, recall, geo list. If None, return a new dictionary.
        '''
        pcs, recall, geo = Metrics.all_metric(pred, label, prnt=prnt)
        if out is None:
            return {'precision': [pcs], 'recall': [recall], 'geo_error': [geo]}
        else:
            return {
                'precision': out['precision'] + [pcs],
                'recall': out['recall'] + [recall],
                'geo_error': out['geo_error'] + [geo]
            }

    @staticmethod
    def all_metric(pred, label, prnt=False):
        # print(pred)
        # print(label)
        pcs = Metrics.precision(pred, label)
        recall = Metrics.recall(pred, label)
        geo = Metrics.geo_error(pred, label)
        if prnt:
            print('Precision=%.2f%%; Recall=%.2f%%; Geographic Error=%.2fm' %
                  (pcs * 100, recall * 100, geo))
        return pcs, recall, geo

    @staticmethod
    def precision(pred, label):
        '''
        # correctly matched segments / pred segments
        25 is the width of road
        :param pred: [[lon], [lat]]
        :param label: [[lon], [lat]]
        :return:
        '''
        pred = [(pred[0][i], pred[1][i]) for i in range(len(pred[0]))]
        label = [(label[0][i], label[1][i]) for i in range(len(label[0]))]
        dists = Metrics.get_all_dist_on_line(label, pred)
        # print(dists)
        tmp = [d < 30 for d in dists]
        print(dists)
        touwei = 0
        if len(tmp) > touwei and sum(tmp[:touwei]) == 0:
            tmp = tmp[touwei:]
        if len(tmp) > touwei and sum(tmp[-touwei:]) == 0:
            tmp = tmp[:-touwei]
        pcs = sum(tmp) / len(dists)
        return pcs

    @staticmethod
    def recall(pred, label):
        '''
        # correctly matched segments / label segments
        :param pred: [[lon], [lat]]
        :param label: [[lon], [lat]]
        :return:
        '''
        pred = [(pred[0][i], pred[1][i]) for i in range(len(pred[0]))]
        label = [(label[0][i], label[1][i]) for i in range(len(label[0]))]
        dists = Metrics.get_all_dist_on_line(pred, label)
        print(dists)
        tmp = [d < 30 for d in dists]
        print(tmp)
        touwei = 0
        print(sum(tmp[:touwei]))
        if len(tmp) > touwei and sum(tmp[:touwei]) == 0:
            tmp = tmp[touwei:]
        if len(tmp) > touwei and sum(tmp[-touwei:]) == 0:
            tmp = tmp[:-touwei]

        recall = sum(tmp) / len(dists)
        return recall

    @staticmethod
    def geo_error(pred, label):
        '''
        # mean of the distance of each pred point to label
        :param pred: [[lon], [lat]]
        :param label: [[lon], [lat]]
        :return:
        '''
        pred = [(pred[0][i], pred[1][i]) for i in range(len(pred[0]))]
        label = [(label[0][i], label[1][i]) for i in range(len(label[0]))]
        dists = Metrics.get_all_dist_on_line(label, pred)
        pcs = np.mean([d if d > 30 else 0 for d in dists])
        return pcs

    @staticmethod
    def get_all_dist_on_line(line, points):
        '''
        Get the distance of each point to the line
        :param line: [(lon, lat)]
        :param points: [(lon, lat)]
        :return:
        '''
        line = LineString(line)
        point_line = LineString(points)
        point_len = point_line.length
        points = [
            point_line.interpolate(x)
            for x in np.linspace(0, point_len, point_len / 0.0001)
        ]
        print('Length:', point_len)
        dists = [
            cal_distance(p.coords[0],
                         line.interpolate(line.project(p)).coords[0])
            for p in points
        ]
        thres, i, max_thres = 30, 0, 80
        dists2 = [1 if x > thres else 0 for x in dists]
        while i < len(dists):
            while i < len(dists) and dists2[i] == 0:
                dists[i] = 0
                i += 1
            j = i
            while j < len(dists) and dists2[j] == 1:
                j += 1
            if sum([k > max_thres for k in dists[i:j]]) == 0:
                for k in range(i, j):
                    dists[k] = 0
            i = j
        return dists


class DTW:
    '''
    Dynamic Time Wrapping Algorithm.
    '''

    def __init__(self, ts1, ts2, is_norm=False):
        '''
        :param ts1: Time series 1 with m time periods. A list [[lon, lat]].
        :param ts2: Time series 2 with n time periods. A list.
        :param is_norm: whether normalization before dtw.
        '''
        if is_norm:
            self.ts1 = self.normalization(ts1)
            self.ts2 = self.normalization(ts2)
        else:
            self.ts1 = ts1
            self.ts2 = ts2
        self.ots1, self.ots2 = ts1, ts2
        self.m = len(ts1)
        self.n = len(ts2)
        self.directions = [[0, 1], [1, 1], [1, 0]]

    def normalization(self, ts):
        ts = np.matrix(ts).T
        m = ts.mean(axis=1)
        std = ts.std(axis=1)
        ts[0] = (ts[0] - m[0]) / std[0]
        ts[1] = (ts[1] - m[1]) / std[1]
        return ts.T.tolist()

    def dtw(self):
        mat = np.ones((self.m, self.n)) + np.inf
        mat_ind = {}
        start_nodes = [(0, 0)]
        mat[start_nodes[0]] = 0
        layer_cnt = 0
        end_node = (self.m - 1, self.n - 1)

        print('m = %d, n = %d' % (self.m, self.n))
        while len(start_nodes) != 0:
            next_nodes = []
            for node in start_nodes:
                for dir in self.directions:
                    succ = (node[0] + dir[0], node[1] + dir[1])
                    if 0 <= succ[0] < self.m and 0 <= succ[1] < self.n:
                        new_dist = mat[node] + cal_distance(
                            self.ts1[succ[0]], self.ts2[succ[1]])
                        if succ in mat_ind.keys():
                            if new_dist < mat[succ]:
                                mat_ind[succ] = [node]
                                mat[succ] = new_dist
                            elif new_dist == mat[succ]:
                                mat_ind[succ].append(node)
                        else:
                            mat_ind[succ] = [node]
                            mat[succ] = new_dist
                        if succ not in next_nodes:
                            next_nodes.append(succ)
            start_nodes = next_nodes
            layer_cnt += 1
            print('Layer %d is finished' % layer_cnt)

        print('Minimum distance = %f' % mat[end_node])

        return mat, mat_ind

    def is_in_bound(self, i, j):
        return 0 <= i < self.m and 0 <= j < self.n

    def dtw_boundary(self):
        '''
        Get a rough upper boundary
        :return:
        '''
        min_dist = 0
        node = [0, 0]
        # for i in range(max(self.m, self.n)-1):
        #     nxt = [node[0] + 1, node[1] + 1]
        #     if nxt[0] >= self.m:
        #         nxt[0] = self.m - 1
        #     if nxt[1] >= self.n:
        #         nxt[1] = self.n - 1
        #     max_dist += cal_distance(self.ts1[nxt[0]], self.ts2[nxt[1]])
        #     node = nxt
        while node != [self.m - 1, self.n - 1]:
            value = np.inf
            for dir in self.directions:
                tmp = [node[0] + dir[0], node[1] + dir[1]]
                if not self.is_in_bound(tmp[0], tmp[1]):
                    continue
                tmp_value = cal_distance(self.ts1[tmp[0]], self.ts2[tmp[1]])
                if tmp_value < value:
                    nxt = tmp
                    value = tmp_value
            min_dist += value
            node = nxt

        return min_dist

    def get_dist_matrix(self, ts1, ts2):
        dist_mat = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                dist_mat[i, j] = cal_distance(ts1[i], ts2[j])
        return dist_mat

    @staticmethod
    def get_derivatives(ts):
        ts.insert(0, ts[0])
        ts.append(ts[-1])
        dts = [[(ts[i + 1][0] - ts[i - 1][0]) / 2,
                (ts[i + 1][1] - ts[i - 1][1]) / 2]
               for i in range(1,
                              len(ts) - 1)]
        return dts

    def get_derivatives_dist_matrix(self, ts1, ts2):
        dts1 = self.get_derivatives(ts1)
        dts2 = self.get_derivatives(ts2)
        return self.get_dist_matrix(dts1, dts2)

    def quick_dtw(self, is_derivatives=False):
        mat = np.ones((self.m, self.n)) + np.inf
        if is_derivatives:
            dist_mat = self.get_derivatives_dist_matrix(self.ts1, self.ts2)
        else:
            dist_mat = self.get_dist_matrix(self.ts1, self.ts2)
        mat_ind = {}
        start_nodes = [(0, 0)]
        end_node = (self.m - 1, self.n - 1)
        mat[start_nodes[0]] = 0
        layer_cnt = 0

        max_dist = self.dtw_boundary()

        print('m = %d, n = %d, max_dist = %f' % (self.m, self.n, max_dist))
        while len(start_nodes) != 0:
            next_nodes = []
            for node in start_nodes:
                if max_dist < mat[node] < np.inf:
                    continue
                for dir in self.directions:
                    succ = (node[0] + dir[0], node[1] + dir[1])
                    if 0 <= succ[0] < self.m and 0 <= succ[1] < self.n:
                        new_dist = mat[node] + dist_mat[succ]
                        if new_dist > max_dist:
                            continue
                        if succ in mat_ind.keys():
                            if new_dist < mat[succ]:
                                mat_ind[succ] = [node]
                                mat[succ] = new_dist
                            elif new_dist == mat[succ]:
                                mat_ind[succ].append(node)
                        else:
                            mat_ind[succ] = [node]
                            mat[succ] = new_dist
                        if succ not in next_nodes:
                            next_nodes.append(succ)
            start_nodes = next_nodes
            layer_cnt += 1
            if len(mat[mat != np.inf]) != 0:
                print('Layer %d is finished. Max = %f' %
                      (layer_cnt, mat[mat != np.inf].max()))

        print('Minimum distance = %f' % mat[end_node])

        return mat, mat_ind

    def shortest_path_dtw(self, is_derivatives=False):
        '''
        Solving DTW with shortest path algorithm
        :param is_derivatives:
        :return:
        '''
        mat = np.ones((self.m, self.n)) + np.inf
        if is_derivatives:
            dist_mat = self.get_derivatives_dist_matrix(self.ts1, self.ts2)
        else:
            dist_mat = self.get_dist_matrix(self.ts1, self.ts2)
        mat_ind = {}
        mat[0, 0] = 0

        for layer in range(1, self.m + self.n):
            for i in range(layer + 1):
                node = (i, layer - i)
                if not self.is_in_bound(node[0], node[1]):
                    continue
                dist = dist_mat[node]
                for dir in self.directions:
                    lst_node = (node[0] - dir[0], node[1] - dir[1])
                    if not self.is_in_bound(lst_node[0], lst_node[1]):
                        continue
                    tmp_value = mat[lst_node] + dist
                    if tmp_value < mat[node]:
                        mat[node] = tmp_value
                        mat_ind[node] = [lst_node]
                    elif tmp_value == mat[node]:
                        mat_ind[node].append(lst_node)

        # print('Minimum distance = %f' % mat[self.m-1, self.n-1])

        return mat, mat_ind

    def get_trace(self, mat_ind):
        '''
        Now only get one best trace
        :param mat_ind:
        :return:
        '''
        trace = [(self.m - 1, self.n - 1)]
        node = trace[0]
        while node != (0, 0):
            last_node = mat_ind[node][0]
            trace.append(last_node)
            node = last_node

        # Print trace
        trace_mat = np.zeros((self.m, self.n))
        for t in trace:
            trace_mat[t[0], t[1]] = 1
        # print(trace_mat)

        return trace, trace_mat

    def get_new_ts(self, mat_ind=None):
        if mat_ind is None:
            _, mat_ind = self.shortest_path_dtw()
        trace, _ = self.get_trace(mat_ind)
        ts1, ts2 = self.ts1, self.ts2
        nts1, nts2 = [ts1[0]], [ts2[0]]
        for t in trace[::-1]:
            nts1.append(ts1[t[0]])
            nts2.append(ts2[t[1]])
        return nts1, nts2

    def get_average_dist(self, mat_ind=None):
        if mat_ind is None:
            _, mat_ind = self.shortest_path_dtw()
        nts1, nts2 = self.get_new_ts(mat_ind)
        dist = 0
        for t in range(len(nts1)):
            dist += cal_distance(nts1[t], nts2[t])
        return dist / len(nts1)

    def get_median_dist(self, mat_ind=None):
        if mat_ind is None:
            _, mat_ind = self.shortest_path_dtw()
        nts1, nts2 = self.get_new_ts(mat_ind)
        dist = []
        for t in range(len(nts1)):
            dist.append(cal_distance(nts1[t], nts2[t]))
        return np.median(dist)

    def get_middle_ts(self, mat_ind=None):
        nts1, nts2 = self.get_new_ts(mat_ind)
        return ((np.array(nts1) + np.array(nts2)) / 2).tolist()

    def plot(self, trace, title='DTW'):
        ts1 = np.matrix(self.ots1).T.tolist()
        ts2 = np.matrix(self.ots2).T.tolist()
        plt.title(title)
        plt.plot(ts1[0], ts1[1], color='blue', linewidth="0.8")
        plt.plot(ts2[0], ts2[1], color='red', linewidth="0.8")
        for line in trace:
            p1 = self.ots1[line[0]]
            p2 = self.ots2[line[1]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                     color='black',
                     linewidth="0.4")
        plt.grid(True, linestyle="--", color="black", linewidth="0.4")
        plt.savefig('../../res/work/0401_ddtw/pic/%s.png' % title)
        plt.show()


class EvaluateMetric:
    def __init__(self):
        self.pcs = []
        self.recall = []
        self.geo_error = []

    def update_metric(self, pred, label):
        pcs, recall, geo_error = Metrics.all_metric(pred, label)
        self.pcs.append(pcs)
        self.recall.append(recall)
        self.geo_error.append(geo_error)


def get_date_type(date0, no_decimal=False):
    '''
    :param date0: yyyy/mm/dd HH:MM:SS.xxx
    :param no_decimal: If True: yyyy/mm/dd HH:MM:SS.0
    :return A datetime date
    Transfer a string time to datetime format.
    Specifically for signal data reading.
    '''
    day, time = date0.split(' ')
    day = day.split('/')
    time = time.split(':')
    return datetime.datetime(int(day[0]), int(day[1]), int(day[2]),
                             int(time[0]), int(time[1]), int(float(time[2])),
                             0 if no_decimal else int(float(time[2]) % 1))


def check_dirs(dirs):
    if type(dirs) == list:
        for d in dirs:
            check_dir(d)
    else:
        check_dir(dirs)


def check_dir(d):
    if d == '':
        return
    elif not os.path.exists(d):
        check_dir(os.path.dirname(d))
        os.mkdir(d)
    else:
        return


def back_up(file_path):
    '''Back up the file with the end "_back" in the same directory.'''
    f = os.path.basename(file_path)
    d = os.path.dirname(file_path)
    shutil.copy(file_path, os.path.join(d, '_back.'.join(f.split('.'))))
    return


def output_data_js(df, output_f, var_name='data'):
    '''
    output the longitude and latitude data as js file
    :param df: A pandas DataFrame, must contain 'lon' and 'lat' columns.
    :param output_f: the filename to output js file
    :param var_name: Variable name in js file
    :return: No return, just write a js file.
    Js File example:
    var lon = 139
    var lat = 39
    var data = {'points': [[139, 39], [139.1, 39.1]]}
    '''
    lon = df.lon.median()
    lat = df.lat.median()
    max_lon = df.lon.max() * 1.5 - df.lon.min() * 0.5
    min_lon = df.lon.min() * 1.5 - df.lon.max() * 0.5
    max_lat = df.lat.max() * 1.5 - df.lat.min() * 0.5
    min_lat = df.lat.min() * 1.5 - df.lat.max() * 0.5
    N = len(df)
    content = '''var lon = %.5f\nvar lat = %.5f\nvar max_lon = %.5f\nvar min_lon = %.5f''' % (
    lon, lat, max_lon, min_lon) \
              + '''\nvar max_lat = %.5f\nvar min_lat = %.5f\nvar \n''' % (max_lat, min_lat) \
              + '''%s = {'points': [''' % var_name
    for i in range(N):
        if 'dates' in df.columns:
            content = '''%s[%.5f, %.5f, '%s']''' % (
                content, df.iloc[i].lon, df.iloc[i].lat, str(df.iloc[i].dates))
        else:
            content = '''%s[%.5f, %.5f]''' % (content, df.iloc[i].lon,
                                              df.iloc[i].lat)
        if i != N - 1:
            content = '''%s, ''' % content
        else:
            content = '''%s]}''' % content
    f = open(output_f, 'w')
    f.write(content)
    f.close()


def log_print(info, type='info', end='\n'):
    print(info, end=end)
    if type == 'error':
        logging.error(info)
    else:
        logging.info(info)


def cal_distance(pos1, pos2):
    '''
    Calculate m from pos1 to pos2
    :param pos1: [lon, lat]
    :param pos2: [lon, lat]
    :return:
    '''
    lon1, lat1, lon2, lat2 = map(radians, [pos1[0], pos1[1], pos2[0], pos2[1]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # earth radius, km
    return c * r * 1000


def cal_direction(pos1, pos2):
    '''
    Calculate angle from pos1 to pos2
    :param pos1:
    :param pos2:
    :return:
    '''
    radLatA = radians(pos1[1])
    radLonA = radians(pos1[0])
    radLatB = radians(pos2[1])
    radLonB = radians(pos2[0])
    dLon = radLonB - radLonA
    y = sin(dLon) * cos(radLatB)
    x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)
    brng = degrees(atan2(y, x))
    brng = (brng + 360) % 360
    return brng


class ShanghaiData:
    '''
    读取上海市收集到的信令数据
    有公交车、私家车、地铁和自行车的数据
    '''

    @staticmethod
    def get_data(raw_cell_dir, sh_type='bus', clean=True):
        raw_cell_dir = os.path.join(raw_cell_dir, sh_type)
        signals = pd.read_csv(
            os.path.join(
                raw_cell_dir,
                '%s_signals%s.csv' % (sh_type, '_clean' if clean else '')))
        gps = pd.read_csv(
            os.path.join(raw_cell_dir, '%s_gps%s.csv' %
                         (sh_type, '_clean' if clean else '')))
        signals.dates = signals.dates.apply(RawSignal.get_datetime)
        gps.dates = gps.dates.apply(RawSignal.get_datetime)
        return signals, gps

    @staticmethod
    def get_multi_data(raw_cell_dir, sh_type='bus'):
        raw_signal, gps = ShanghaiData.get_data(raw_cell_dir, sh_type, True)
        gps_files = np.unique(raw_signal.gps_user_id)
        out = []
        for gps_user_id in gps_files:
            signals = []
            tmp_signal = raw_signal[raw_signal.gps_user_id == gps_user_id]
            for vid in np.unique(tmp_signal.video_id):
                signals.append(
                    tmp_signal[tmp_signal.video_id == vid].reset_index(
                        drop=True))
            out.append([
                signals,
                gps[gps.gps_user_id == gps_user_id].reset_index(drop=True)
            ])
        return out


class CrawlCellular:
    @staticmethod
    def crawl_lac_ci(lac, ci, type='dianxin'):
        if type == 'dianxin':
            mnc = 11
        elif type == 'yidong':
            mnc = 0
        else:
            mnc = 1
        url_f = 'http://v.juhe.cn/cell/query?mnc={}&lac={}&ci={}&hex=&key=ac90fd4c7d4311cbf9de421e6acecae3'
        url = url_f.format(mnc, lac, ci)
        tmp = CrawlCellular.crawl_base(url)
        if tmp['error_code'] == 0:
            res = tmp['result']
            # TODO:
        return None

    @staticmethod
    def crawl_cmda(sid, nid, bid):
        url_f = 'http://v.juhe.cn/cdma/?sid={}&nid={}&cellid={}&hex=&dtype=&callback=&key=7365d57fdae1be309a6d78e65ef4213a'
        url = url_f.format(sid, nid, bid)
        tmp = CrawlCellular.crawl_base(url)
        if tmp['error_code'] == 0:
            res = tmp['result']
            return {
                'lon': float(res['lon']),
                'lat': float(res['lat']),
                'olon': float(res['o_lon']),
                'olat': float(res['o_lat']),
                'address': res['address'],
                'radius': int(res['raggio'])
            }
        return None

    @staticmethod
    def crawl_cmda_df(df, cell_path):
        '''
        :param df: pd.DataFrame with ['sid', 'nid', 'bid']，都是int型
        :param cell_path: 保存cell dict的json文件路径，格式是{'sid_nid_bid': {lon, lat...}}
        :return: cell dict
        '''
        if os.path.exists(cell_path):
            cells = Json.load(cell_path)
        else:
            cells = {}

        df = df[['sid', 'nid', 'bid']]
        df = df.dropna()
        cell_set = list(
            set([(df.sid.iloc[i], df.nid.iloc[i], df.bid.iloc[i])
                 for i in range(len(df))]))

        for c_i, cell in enumerate(cell_set):
            cell_str = '_'.join(list(map(str, cell)))
            if c_i % 500 == 0:
                print('Crawling cells %d/%d ...' % (c_i, len(cell_set)))
            if cell_str not in cells:
                tmp = CrawlCellular.crawl_cmda(*cell)
                cells[cell_str] = tmp

        Json.output(cells, cell_path)
        return cells

    @staticmethod
    def crawl_base(url):
        response = urllib.request.urlopen(url)
        j = response.read().decode('utf-8')
        j = json.loads(j)
        return j

    @staticmethod
    def cmda_json2df(cells):
        '''
        使用纠偏后的坐标
        '''
        out = []
        for k, v in cells.items():
            if 'None' in k: continue
            sid, nid, bid = list(map(int, k.split('_')))
            if v is not None and len(v) > 0:
                out.append([
                    sid, nid, bid, v['lon'], v['lat'], v['radius'],
                    v['address']
                ])
        out = pd.DataFrame(out)
        out.columns = [
            'sid', 'nid', 'bid', 'cmda_lon', 'cmda_lat', 'radius', 'address'
        ]
        for c in ['sid', 'nid', 'bid']:
            out[c] = out[c].astype(str)
        # out.to_csv('data/results/cells.csv', index=False)
        return out


if __name__ == '__main__':
    args = get_utils_args()
    # raw_signal = RawSignal(args.signal_dir)
    # raw_signal.get_signals('20170607', '00000')

    # user_signal = UserSignal('hycRBenKLjyVm1VmA3J9jA==', args.signal_dir)
    # user_signal.get_signal('20170623')

    # alluser_signal = AllUsersSignal(args.signal_dir)
    # alluser_signal.user_daily_signal_count_base('20170623', 'hy', True)
    # alluser_signal.user_daily_signal_count('20170623', plot=False, unique_cell=True)
    # counts = Json.load('/Users/Viki/Documents/yhliu/Smart City/data/hf_signals/user_analysis/user_daily_signal_count(unique)_tmp.json')
    # alluser_signal.plot_user_daily_signal_count(counts['20170623'], '20170623', True)

    # alluser_signal.user_daily_signal_count_dict('20170623', 'hy')
    # alluser_signal.get_different_user_level_user_signals()
    # alluser_signal.show_different_user_level_user_signals()
    # alluser_signal.signal_time_interval('20170623')

    # mymap = MapPlot()
    # MapPlot.plot_traj([[117.23, 117.24]], [[31.81, 31.8]], 12)
    # MapPlot.plot_signal_user('V2gWC8HgBdzTiBAfk/lwcw==',
    #                          "/Users/Viki/Documents/yhliu/Smart City/data/hf_signals/user_based_signals/20170623",
    #                          save_path='/Users/Viki/Documents/yhliu/project/BigEyeOut/tmp/tmp.html')

    # Coordinate().eval_bd_wgs(111, 32)
