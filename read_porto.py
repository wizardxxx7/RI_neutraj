import pandas as pd
import os
import numpy as np


class Porto(object):

    def __init__(self, file_path):
        self.file_path = file_path
        self.datalength = 1710670

    def ll_range(self, name):
        if name not in ['train.csv', 'test.csv']:
            print('Invalid filename!!!')
            return
        data = pd.read_csv(os.path.join(self.file_path, name))
        trajs = data['POLYLINE']
        lons, lats = [], []
        traj = []
        for i, x in enumerate(trajs):
            print(i, i / self.datalength)
            tmp = self.str2traj(x)
            if tmp != None:
                traj.append(tmp)
            else:
                traj.append(np.nan)
            # for y in traj:
            #     lons.append(y[0])
            #     lats.append(y[1])
        data['POLYLINE'] = traj
        data.dropna()
        data.to_csv(os.path.join(self.file_path, 'Train.csv'))
        # return [min(lons), max(lons)], [max(lats), max(lats)]

    def str2traj(self, x):
        traj = x[1:-2]
        traj = traj.split(']')
        ret = [i[1:-1].split(',') if i == 0 else i[2:-1].split(',') for i in traj]
        try:
            ret = [[float(x) for x in y] for y in ret]
        except:
            return None
        return ret

    # def readfile(self, name):
    #     file = pd.read_csv(os.path.join(self.file_path, name))
    #     # print(file.get_chunk(10000))
    #     return file


if __name__ == '__main__':
    file_path = '/home/yiwei/data/Porto'
    f = Porto(file_path)
    lon_range, lat_range = f.ll_range('train.csv')
