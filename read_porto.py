import pandas as pd
import os
import numpy as np


class Porto(object):

    def __init__(self, file_path):
        self.file_path = file_path

    def readfile(self, name):
        if name not in ['train.csv', 'test.csv']:
            print('Invalid filename!!!')
            return

        file = pd.read_csv(os.path.join(self.file_path, name), iterator=True)
        print(file.get_chunk(10000))
        return


if __name__ == '__main__':
    file_path = '/home/yiwei/data/Porto'
    f = Porto(file_path)
    f.readfile('train.csv')
