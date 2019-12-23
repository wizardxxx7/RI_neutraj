from tools import preprocess
from tools.distance_compution import trajectory_distance_combain, trajecotry_distance_list
from geo_rnns.neutraj_trainer import NeuTrajTrainer
from tools import config
import os
import pandas as pd
import  numpy as np
import _pickle as cPickle
import numpy as  np


def distance_comp(coor_path, valid_traj_num, data_name, distance_type='discret_frechet'):
    traj_coord = cPickle.load(open(coor_path, 'rb'))[0]
    np_traj_coord = []
    for t in traj_coord:
        np_traj_coord.append(np.array(t))
    print(np_traj_coord[0])
    print(np_traj_coord[1])
    print(len(np_traj_coord))

    batch_size = int(valid_traj_num / 10)
    trajecotry_distance_list(np_traj_coord, batch_size=batch_size, processors=15, distance_type=distance_type,
                             data_name=data_name)

    trajectory_distance_combain(valid_traj_num, batch_size=batch_size,
                                metric_type=distance_type, data_name=data_name)


if __name__ == '__main__':

    # Preprocessing
    coor_path, data_name, valid_traj_num = preprocess.trajectory_feature_generation(path='./data/toy_trajs')
    distance_list = ["sspd", "dtw", "lcss", "hausdorff", "frechet", "discret_frechet", "sowd_grid", "erp", "edr"]

    # for distance_type in distance_list:
    distance_type = 'discret_frechet'
    distance_comp(coor_path, valid_traj_num, data_name, distance_type)

    # # Training
    # trajrnn = NeuTrajTrainer(tagset_size=config.d, batch_size=config.batch_size,
    #                          sampling_num=config.sampling_num)
    # trajrnn.data_prepare(griddatapath=config.gridxypath, coordatapath=config.corrdatapath,
    #                      distancepath=config.distancepath, train_radio=config.seeds_radio)
    # model_name = trajrnn.neutraj_train(load_model=None, in_cell_update=config.incell,save_model=True,
    #                       stard_LSTM=config.stard_unit)
    #
    # # Testing
    #
    # acc = trajrnn.trained_model_eval(load_model=model_name,in_cell_update=True, stard_LSTM=False)
    # print (acc)
    # res1 = {'HR@10': acc[0], 'HR@50':acc[1], 'R10@50':acc[2], 'Error_true':acc[3], 'Error_test':acc[4], 'Error_div':acc[5]}












'''
    1. 'sspd'

        Computes the distances using the Symmetrized Segment Path distance.

    2. 'dtw'

        Computes the distances using the Dynamic Path Warping distance.

    3. 'lcss'

        Computes the distances using the Longuest Common SubSequence distance

    4. 'hausdorf'

        Computes the distances using the Hausdorff distance.

    5. 'frechet'

        Computes the distances using the Frechet distance.

    6. 'discret_frechet'

        Computes the distances using the Discrete Frechet distance.

    7. 'sowd_grid'

        Computes the distances using the Symmetrized One Way Distance.

    8. 'erp'

        Computes the distances using the Edit Distance with real Penalty.

    9. 'edr'

        Computes the distances using the Edit Distance on Real sequence.
'''
