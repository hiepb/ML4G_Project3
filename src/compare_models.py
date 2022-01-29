from original_model import Graph_OurConvNet
from pyg_model import RGGConvModel
import block

import argparse
import pickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='The task for wich the task/model/optimizer parameters will be set', type=str, choices=['matching', 'clustering'])
    return parser.parse_args()

def main():
    args = parse_args()
    if args.task == 'matching':
        task_parameters = {}
        task_parameters['flag_task'] = 'matching'
        task_parameters['nb_communities'] = 10
        task_parameters['nb_clusters_target'] = 2
        task_parameters['Voc'] = 3
        task_parameters['size_min'] = 15
        task_parameters['size_max'] = 25
        task_parameters['size_subgraph'] = 20
        task_parameters['p'] = 0.5
        task_parameters['q'] = 0.1
        task_parameters['W0'] = block.random_graph(task_parameters['size_subgraph'],task_parameters['p'])
        task_parameters['u0'] = np.random.randint(task_parameters['Voc'],size=task_parameters['size_subgraph'])
        file_name = '/home/userw/ml4g/project03/spatial_graph_convnets/data/set_100_subgraphs_p05_size20_Voc3_2017-10-31_10-23-00_.txt'
        with open(file_name, 'rb') as fp:
            all_trainx = pickle.load(fp)
        task_parameters['all_trainx'] = all_trainx[:100]
    elif args.task == 'clustering':
        task_parameters = {}
        task_parameters['flag_task'] = 'clustering'
        task_parameters['nb_communities'] = 10
        task_parameters['nb_clusters_target'] = task_parameters['nb_communities']
        task_parameters['Voc'] = task_parameters['nb_communities'] + 1
        task_parameters['size_min'] = 5
        task_parameters['size_max'] = 25
        task_parameters['p'] = 0.5
        task_parameters['q'] = 0.1
        file_name = '/home/userw/ml4g/project03/spatial_graph_convnets/data/set_100_clustering_maps_p05_q01_size5_25_2017-10-31_10-25-00_.txt'
        with open(file_name, 'rb') as fp:
            all_trainx = pickle.load(fp)
        task_parameters['all_trainx'] = all_trainx[:100]
    else:
        raise NameError(f'Task {args.task} is not defined')


    # network parameters
    net_parameters = {}
    net_parameters['Voc'] = task_parameters['Voc']
    net_parameters['D'] = 50
    net_parameters['nb_clusters_target'] = task_parameters['nb_clusters_target']
    net_parameters['H'] = 50
    net_parameters['L'] = 10
    #custom
    net_parameters['flag_task'] = task_parameters['flag_task']


    # optimization parameters
    opt_parameters = {}
    opt_parameters['learning_rate'] = 0.00075   # ADAM
    opt_parameters['max_iters'] = 5000
    opt_parameters['batch_iters'] = 100
    if 2==1: # fast debugging
        opt_parameters['max_iters'] = 101
        opt_parameters['batch_iters'] = 10
    opt_parameters['decay_rate'] = 1.25

    dim_in = net_parameters['Voc']
    dim_emb = net_parameters['D']
    dim_hid = net_parameters['H']
    dim_out = net_parameters['nb_clusters_target']
    num_hidLayers = net_parameters['L']


    print('\n\nORIGINAL MODEL################################################')
    orgModel = Graph_OurConvNet(net_parameters)
    print(orgModel)


    print('\n\nPYTORCH GEOMETRIC MODEL#######################################')
    pygModel = RGGConvModel(dim_in, dim_emb, dim_hid, dim_out, num_hidLayers)
    print(pygModel)

    print('\n\nNETWORK PARAMETERS############################################')
    translate = dict()
    translate['Voc'] = 'input dimension'
    translate['D'] = 'embedding dimnesion'
    translate['H'] = 'hidden layer dimension'
    translate['nb_clusters_target'] = 'output dimension'
    translate['L'] = 'nuber of ResGatedGraphConv/OurConvNetcell layers'
    translate['flag_task'] = net_parameters['flag_task']
    for key, value in net_parameters.items():
        print(f'{key}({translate[key]}): {value}')

if __name__ == '__main__':
    main()
