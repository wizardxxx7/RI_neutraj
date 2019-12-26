from tools import preprocess
from tools.distance_compution import trajectory_distance_combain, trajecotry_distance_list
from utils import cal_distance
from geo_rnns.neutraj_trainer import NeuTrajTrainer
from tools import config
import os
import pandas as pd
import numpy as np
import _pickle as cPickle
import numpy as  np
from geolife import geolife
from tensorboardX import SummaryWriter


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
    # Preprocessing Part
    data_list = ['toy_trajs', 'Porto', 'geolife']
    data_type = data_list[2]

    file_path = '/home/yiwei/data/Geolife1.3/Data'
    f = geolife(file_path)
    grid_size = [1100, 1100]

    coor_path = './features/{}/{}_traj_coord'.format(data_type, data_type)
    grid_path = './features/{}/{}_traj_grid'.format(data_type, data_type)

    valid_traj_num = preprocess.trajectory_feature_generation_geolife(f=f, path='./data/' + data_type,
                                                                      lat_range=f.lat_range, lon_range=f.lon_range,
                                                                      min_length=50)
    # valid_traj_num = 2364
    distance_list = ["dtw", "discret_frechet", "hausdorff", "erp", "sspd", "lcss", "frechet", "sowd_grid", "edr"]

    for i in range(3):
        # Training Part
        distance_type = distance_list[i]
        # distance_type = 'hausdorff'
        distance_comp(coor_path, valid_traj_num, data_type, distance_type)
        distance_path = './features/' + data_type + '/' + data_type + '_' + distance_type + '_distance_all_' + str(
            valid_traj_num)
        if distance_type == 'dtw':
            mail_pre_degree = 16
        else:
            mail_pre_degree = 8

        embed_dim, batch_size, sampling_num = 128, int(valid_traj_num * config.seeds_radio / 20), 10

        trajrnn = NeuTrajTrainer(tagset_size=embed_dim, batch_size=batch_size, sampling_num=sampling_num,
                                 data_type=data_type, distance_type=distance_type, embed_dim=embed_dim,
                                 datalength=valid_traj_num, grid_size=grid_size)

        trajrnn.data_prepare(griddatapath=grid_path, coordatapath=coor_path,
                             distancepath=distance_path, train_radio=config.seeds_radio)
        load_model_name = None

        model_list, train_time = trajrnn.neutraj_train(mail_pre_degree=mail_pre_degree, load_model=None,
                                                       in_cell_update=config.incell, save_model=True,
                                                       stard_LSTM=config.stard_unit, test_num=100)
        # Evaluation Part
        eval_list = []
        # model_list = ['./model/Porto_dtw_GRU_training_2214_19_incellTrue_config_False_0.005_22_10_0.2_Porto_False_128_train_0.64_test_0.7336_0.973_1325.22589120676_1546.245478418305_221.01958721154494.h5']
        for model in model_list:
            ret = trajrnn.trained_model_eval(mail_pre_degree=mail_pre_degree, test_num=1700, load_model=model,
                                             in_cell_update=True, stard_LSTM=False)
            acc, embedding_time = ret[0], ret[1]
            # res = {'HR@10': acc[0], 'HR@50': acc[1], 'R10@50': acc[2],  # 'Error_true': acc[3], 'Error_test': acc[4],
            #        'Error_div': acc[5], 'Avg_search_time': acc[6], 'Embed_time': embedding_time}
            res = [acc[0], acc[1], acc[2], acc[5], acc[6], embedding_time]
            eval_list.append(res)

        for i in eval_list:
            print(i)
        run_path = 'runs/' + data_type + '_' + distance_type
        writer = SummaryWriter(run_path)
        for i in range(len(eval_list)):
            writer.add_scalar('HR@10', eval_list[i][0], global_step=i)
            writer.add_scalar('HR@50', eval_list[i][1], global_step=i)
            writer.add_scalar('R10@50', eval_list[i][2], global_step=i)
            writer.add_scalar('Error_div', eval_list[i][3], global_step=i)
            writer.add_scalar('Avg_search_time', eval_list[i][4], global_step=i)
            writer.add_scalar('Embed_time', eval_list[i][5], global_step=i)

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
