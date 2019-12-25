import pandas as pd
import os
import numpy as np
from utils import MapPlot, Coordinate, Json

import time
MARK_COLOR = [
    'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue',
    'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen',
    'gray', 'black', 'lightgray'
]


class Porto(object):

    def __init__(self, file_path):
        self.file_path = file_path
        self.lon_range = [-8.6560425, -8.5864875]
        self.lat_range = [41.13912, 41.2095]


    def read_data(self, name = 'train.json'):
        self.data = Json.load(os.path.join(self.file_path, name))

    def generate_json(self, name, out_name, sample=82):
        if name not in ['train.csv', 'test.csv']:
            print('Invalid filename!!!')
            return
        data = pd.read_csv(os.path.join(self.file_path, name))
        trajs = data['POLYLINE']
        cnt = 0
        trip = {}
        for i, x in enumerate(trajs):
            if i % sample != 0:
                continue
            tmp = self.str2traj(x)
            tr = self.judge(tmp)
            if len(tr) > 0:
                trip[str(cnt)] = [[float(tr[j][0]), float(tr[j][1]), int(data.TIMESTAMP.iloc[i] + 15 * j)] for j in range(len(tr))]
                cnt += 1
            print (i, cnt)

        Json.output(trip, os.path.join(self.file_path, out_name))
        return

    def judge(self, traj, num=10):
        '''
        1. out of lon_lat range
        2. less than num gps points
        '''
        ret = []
        for x in traj:
            if x[0] >= self.lon_range[0] and x[0] <= self.lon_range[1] \
                    and x[1] >= self.lat_range[0] and x[1] <= self.lat_range[1]:
                ret.append(x)
            else:
                return []
        if len(ret) >= num:
            return ret
        else:
            return []

    def str2traj(self, x):
        traj = x[1:-2]
        traj = ',' + traj
        traj = traj.split(']')
        ret = [i.split(',') for i in traj]
        try:
            ret = [[float(y[1][1:]), float(y[2])] for y in ret]
        except:
            return []
        return ret

    def get_range(self, ):
        trajs = self.test_data['POLYLINE']
        lons, lats = [], []
        for i, x in enumerate(trajs):
            print(i, len(trajs))
            for j in x:
                lons.append(j[0])
                lats.append(j[1])
        trajs = self.train_data['POLYLINE']
        for i, x in enumerate(trajs):
            try:
                print(i, len(trajs))
                for j in x:
                    lons.append(j[0])
                    lats.append(j[1])
            except:
                pass

        return max(lons), max(lats), min(lons), min(lats)


if __name__ == '__main__':
    file_path = '/home/yiwei/data/Porto'
    f = Porto(file_path)
    # f.read_data('test.json')
    f.generate_json('train.csv', 'small.json')
    # s = time.time()
    # js = Json.load(os.path.join(f.file_path, 'test.json'))
    # print (time.time() - s)
    # print (len(js))
    # f.gef_train_data()
    # f.gef_test_data()
    # res = f.get_range()
    # c = Coordinate()
    # trajs = f.test_data['POLYLINE']
    # lons = [[y[0] for y in x] for x in trajs]
    # lats = [[y[1] for y in x] for x in trajs]
    # MapPlot.plot_traj(lons, lats, transforms=False)
