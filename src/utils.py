import graph_generator

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def old_to_pyg_data(graph):
    x = graph.signal
    E_start = graph.edge_to_starting_vertex
    E_end = graph.edge_to_ending_vertex

    num_edges = E_start.getnnz()
    E_start = torch.from_numpy(E_start.toarray()).type(torch.float)
    E_end = torch.from_numpy(E_end.toarray()).type(torch.float)
    edge_index = []
    for i in range(num_edges):
        edgeStart = E_start[i]
        nodeStart = torch.argmax(edgeStart)

        edgeEnd = E_end[i]
        nodeEnd = torch.argmax(edgeEnd)
        edge_index.append([nodeStart, nodeEnd])

    edge_index = torch.tensor(edge_index).T
    print(edge_index)


def to_pyg_data(graph):
    x = graph.signal

    tensor_adj_matrix = torch.from_numpy(graph.adj_matrix)
    edge_index = (tensor_adj_matrix > 0).nonzero().t()

    y = graph.target

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def compute_loss_weight(y):
    # compute loss weigth
    labels = y.detach().clone().cpu().numpy()
    V = labels.shape[0]
    nb_classes = len(np.unique(labels))
    cluster_sizes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(labels == r)[0]
        cluster_sizes[r] = len(cluster)
    weight = torch.zeros(nb_classes)
    for r in range(nb_classes):
        sumj = 0
        for j in range(nb_classes):
            if j != r:
                sumj += cluster_sizes[j]
        weight[r] = sumj / V
    return weight


def compute_confusion_matrix(out, y):
    """Compute the average of the diagonal of the normalized confusion
    matrix w.r.t. the cluster sizes (the confusion matrix measures the number
    of nodes correctly and badly classified for each class).
    """
    S = y.detach().clone().cpu().numpy()
    # C = np.argmax( torch.nn.Softmax(dim=0)(y).data.cpu().numpy() , axis=1)
    C = np.argmax(torch.nn.Softmax(dim=0)(out).detach().clone().cpu().numpy(), axis=1)
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    train_y = y.detach().clone().cpu().numpy()
    for r in range(nb_classes):
        cluster = np.where(train_y == r)[0]
        CM[r, :] /= cluster.shape[0]
    return CM, nb_classes


def create_plots(data, xlabel, path):
    print(data)
    x = []
    y_acc = []
    y_time_train = []
    y_time_test = []
    for key in data.keys():
        x.append(key)
        y_acc.append(data[key][0])
        y_time_train.append(data[key][1])
        y_time_test.append(data[key][2])

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y_acc)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy')
    plt.savefig(path+'_accuracy')

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y_time_train)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Batch time train (sec)')
    plt.savefig(path+'_time_train')

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y_time_test)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Batch time test (sec)')
    plt.savefig(path+'_time_test')


if __name__ == '__main__':
    import pickle
    import block

    from collections import OrderedDict

    data = OrderedDict([(1, (0.7410000044107437, 0.3379010558128357, 0.3219701051712036)), (2, (0.9604166674613952, 0.44071486592292786, 0.35011231899261475)), (3, (0.9636250057816506, 0.542947381734848, 0.38097822666168213)), (4, (0.9975000014901161, 0.6440851092338562, 0.41156005859375)), (5, (0.9972083348035812, 0.7463850975036621, 0.4456031322479248)), (6, (0.9942500033974648, 0.8445881605148315, 0.47296643257141113)), (7, (0.9944583350419998, 0.9684409201145172, 0.5133578777313232)), (8, (0.9956250035762787, 1.088496595621109, 0.5536187887191772)), (9, (0.9950833368301392, 1.1776520013809204, 0.5735207796096802)), (10, (0.9982916679978371, 1.2905015647411346, 0.6090717315673828))])
    create_plots(data, 'test', 'testplot')
    exit()

    def print_graph_info(graph):
        print(f'adjacency matrix ({type(graph.adj_matrix)}):\n {graph.adj_matrix}')
        print(f'edge_to_starting_vertex ({type(graph.edge_to_starting_vertex)}):\n {graph.edge_to_starting_vertex}')
        print(f'edge_to_ending_vertex ({type(graph.edge_to_ending_vertex)}):\n {graph.edge_to_ending_vertex}')
        print(f'signal ({type(graph.signal)}):\n {graph.signal}')
        print(f'target ({type(graph.target)}):\n {graph.target}')

    # matching task_parameters
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

    graph_matching = graph_generator.variable_size_graph(task_parameters)

    print_graph_info(graph_matching)

    data = to_pyg_data(graph_matching)
    print(data)

    # old_to_pyg_data(graph_matching)

    exit()



    # clustering task_parameters
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


    graph_clustering = graph_generator.graph_semi_super_clu(task_parameters)
