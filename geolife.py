import pandas as pd
import os
import numpy as np
from utils import MapPlot, Coordinate, Json
import datetime

import time


class geolife(object):
    def __init__(self, filepath):
        self.file_path = filepath
        self.lat_range = [39.6, 40.7]
        self.lon_range = [115.9, 117.1]

    def read_data(self, name='all.json'):
        self.data = Json.load(os.path.join(self.file_path, name))

    def generate_json_single(self, path):
        f = open(path, 'r')
        ret, tmp = [], []
        for k, v in enumerate(f.readlines()):

            if k > 5:
                a = v.split(',')
                strtime = a[5] + ' ' + a[6].strip()
                timestamp = datetime.datetime.strptime(strtime, '%Y-%m-%d %H:%M:%S')
                timestamp += datetime.timedelta(hours=8)
                # print(k, a[0], a[1], timestamp)
                if k == 6: pre = timestamp
                if k > 6 and timestamp - pre > datetime.timedelta(seconds=1200):
                    ret.append(tmp)
                    tmp = []

                pre = timestamp
                tmp.append([str(a[1]), str(a[0]), timestamp.strftime('%Y-%m-%d %H:%M:%S')])
        # print(ret)
        return ret

    def generate_json_all(self, name='all.json'):
        ret = {}
        users = os.listdir(self.file_path)
        cnt = 0
        for i in range(len(users)):  # 000
            print(i, len(ret))
            username = users[i]
            traj_path = os.path.join(self.file_path, username, 'Trajectory')
            trajs = os.listdir(traj_path)
            for j in range(len(trajs)):
                ret_traj = self.generate_json_single(os.path.join(traj_path, trajs[j]))
                for k in ret_traj:
                    if len(k) < 10:
                        continue
                    ret[str(cnt)] = k
                    cnt += 1
                    # print(ret[str(cnt)])
        Json.output(ret, os.path.join(self.file_path, name))
        return len(ret)


if __name__ == '__main__':
    file_path = '/home/yiwei/data/Geolife1.3/Data'
    # f = geolife(file_path)
    # f.generate_json_all('all.json')
    dic = {}
    cnt = 0
    js = Json.load('/home/yiwei/data/Geolife1.3/Data/small.json')
    for k, v in js.items():
        if (int(k) % 10 != 0 ):
            print(k)
            dic[str(cnt)] = [[float(x[0]), float(x[1]), x[2]] for x in v]
            cnt += 1
    Json.output(dic, '/home/yiwei/data/Geolife1.3/Data/small.json')
