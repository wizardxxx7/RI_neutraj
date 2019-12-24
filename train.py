from geo_rnns.neutraj_trainer import NeuTrajTrainer
from tools import config
import os
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    print('os.environ["CUDA_VISIBLE_DEVICES"]= {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print(config.config_to_str())

    trajrnn = NeuTrajTrainer(tagset_size=config.d, batch_size=config.batch_size,
                             sampling_num=config.sampling_num)
    trajrnn.data_prepare(griddatapath=config.gridxypath, coordatapath=config.corrdatapath,
                         distancepath=config.distancepath, train_radio=config.seeds_radio)
    load_model_name = None

    model_list, train_time = trajrnn.neutraj_train(load_model=None, in_cell_update=config.incell, save_model=True,
                                                   stard_LSTM=config.stard_unit)

    eval_list = []
    for model in model_list:
        ret = trajrnn.trained_model_eval(test_num=1500, load_model=model, in_cell_update=True, stard_LSTM=False)
        acc, embedding_time = ret[0], ret[1]
        # res = {'HR@10': acc[0], 'HR@50': acc[1], 'R10@50': acc[2],  # 'Error_true': acc[3], 'Error_test': acc[4],
        #        'Error_div': acc[5], 'Avg_search_time': acc[6], 'Embed_time': embedding_time}
        res = acc.append(embedding_time)
        eval_list.append(res)

    for i in eval_list:
        print(i)

    writer = SummaryWriter('runs/Toy_Frechet')
    for i in range(len(eval_list)):
        writer.add_scalar('HR@10', res[i][0], global_step=i)
        writer.add_scalar('HR@50', res[i][1], global_step=i)
        writer.add_scalar('R10@50',res[i][2], global_step=i)
        writer.add_scalar('Error_div', res[i][3], global_step=i)
        writer.add_scalar('Avg_search_time', res[i][4], global_step=i)
        writer.add_scalar('Embed_time', res[i][5], global_step=i)

