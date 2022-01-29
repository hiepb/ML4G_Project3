from original_model import Graph_OurConvNet
from pyg_model import RGGConvModel
import block
import graph_generator as g
import utils

import os
import time
import pickle
import argparse
import numpy as np
import random
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import confusion_matrix
# pip freeze > requirements.txt

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


# # network parameters
net_parameters = {}
net_parameters['Voc'] = task_parameters['Voc']
net_parameters['D'] = 50
net_parameters['nb_clusters_target'] = task_parameters['nb_clusters_target']
net_parameters['H'] = 50
net_parameters['L'] = 10

# custom
net_parameters['flag_task'] = task_parameters['flag_task']
# #print(net_parameters)


# # optimization parameters
opt_parameters = {}
opt_parameters['learning_rate'] = 0.00075   # ADAM
opt_parameters['max_iters'] = 5000
opt_parameters['batch_iters'] = 100
if 2==1: # fast debugging
    opt_parameters['max_iters'] = 101
    opt_parameters['batch_iters'] = 10
opt_parameters['decay_rate'] = 1.25


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iters', type=int, default=5000)
    # parser.add_argument('--nb_communities')
    parser.add_argument('--batch_iters', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.00075)
    parser.add_argument('--no_cuda', help='Set this if cuda is availbable, but you do NOT want to use it.', action='store_true')
    return parser.parse_args()

def main():
    set_seed(42)
    args = parse_args()
    print(args)
    use_cuda = False
    if torch.cuda.is_available() and not args.no_cuda:
        use_cuda = True
        print('Using CUDA: TRUE')
    else:
        print('Using CUDA: FALSE')

    modelOrg = Graph_OurConvNet(net_parameters)
    print(modelOrg)

    dim_in = net_parameters['Voc']
    dim_emb = net_parameters['D']
    dim_hid = net_parameters['H']
    dim_out = net_parameters['nb_clusters_target']
    num_hidLayers = net_parameters['L']
    modelPyg = RGGConvModel(dim_in, dim_emb, dim_hid, dim_out, num_hidLayers)
    print(modelPyg)

    if use_cuda:
        modelOrg = modelOrg.cuda()
        modelPyg = modelPyg.cuda()

    model = modelPyg  # TODO

    if use_cuda:
        model = model.cuda()


    # number of network parameters
    nb_param = 0
    for param in model.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('nb_param=', nb_param, ' L=', net_parameters['L'])


    # optimization parameters
    learning_rate = args.learning_rate
    max_iters = args.max_iters
    batch_iters = args.batch_iters
    decay_rate = opt_parameters['decay_rate']

    # Optimizer
    global_lr = learning_rate
    global_step = 0
    lr = learning_rate
    optimizer = model.update(lr)

    t_start = time.time()
    t_start_total = time.time()
    average_loss_old = 1e10
    running_loss = 0.0
    running_total = 0
    running_conf_mat = 0
    running_accuracy = 0
    tab_results = []

    for iteration in range(max_iters):
        # Edge = start vertex to end vertex
        # E_start = E x V mapping matrix from edge index to corresponding start vertex
        # E_end = E x V mapping matrix from edge index to corresponding end vertex
        graphOrg = g.variable_size_graph(task_parameters)
        graphPyg = utils.to_pyg_data(graphOrg)

        if use_cuda:
            graphPyg = graphPyg.cuda()
        out = model(graphPyg.x, graphPyg.edge_index)

        labels = graphPyg.y.detach().clone().cpu().numpy() # train_y.data.cpu().numpy()
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

        # loss = net.loss(y, train_y, weight)
        if use_cuda:
            weight = weight.cuda()
        loss = model.loss(out, graphPyg.y, weight)
        loss_train = loss.item()
        running_loss += loss_train
        running_total += 1

        # confusion matrix
        # S = train_y.data.cpu().numpy()
        S = graphPyg.y.detach().clone().cpu().numpy()
        # C = np.argmax( torch.nn.Softmax(dim=0)(y).data.cpu().numpy() , axis=1)
        C = np.argmax(torch.nn.Softmax(dim=0)(out).detach().clone().cpu().numpy(), axis=1)
        CM = confusion_matrix(S, C).astype(np.float32)
        nb_classes = CM.shape[0]
        train_y = graphPyg.y.detach().clone().cpu().numpy()
        for r in range(nb_classes):
            cluster = np.where(train_y == r)[0]
            CM[r, :] /= cluster.shape[0]
        running_conf_mat += CM
        running_accuracy += np.sum(np.diag(CM)) / nb_classes

        # backward, update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # learning rate, print results
        if not iteration % batch_iters:

            # time
            t_stop = time.time() - t_start
            t_start = time.time()

            # confusion matrix
            average_conf_mat = running_conf_mat / running_total
            running_conf_mat = 0

            # accuracy
            average_accuracy = running_accuracy / running_total
            running_accuracy = 0

            # update learning rate
            average_loss = running_loss / running_total
            if average_loss > 0.99 * average_loss_old:
                lr /= decay_rate
            average_loss_old = average_loss
            optimizer = model.update_learning_rate(optimizer, lr)
            running_loss = 0.0
            running_total = 0

            # save intermediate results
            tab_results.append([iteration, average_loss, 100 * average_accuracy, time.time()-t_start_total])

            # print results
            if 1==1:
                print('\niteration= %d, loss(%diter)= %.3f, lr= %.8f, time(%diter)= %.2f' %
                      (iteration, batch_iters, average_loss, lr, batch_iters, t_stop))
                #print('Confusion matrix= \n', 100* average_conf_mat)
                print('accuracy= %.3f' % (100 * average_accuracy))


    ############
    # Evaluation on 100 pre-saved data
    ############
    running_loss = 0.0
    running_total = 0
    running_conf_mat = 0
    running_accuracy = 0
    for iteration in range(100):

        # generate one data
        graphOrg = g.variable_size_graph(task_parameters)
        graphPyg = utils.to_pyg_data(graphOrg)

        if use_cuda:
            graphPyg = graphPyg.cuda()

        # forward, loss
        out = model.forward(graphPyg.x, graphPyg.edge_index)
        # y = net.forward(train_x)
        # compute loss weigth
        # labels = train_y.data.cpu().numpy()
        labels = graphPyg.y.detach().clone().cpu().numpy()
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
        if use_cuda:
            weight = weight.cuda()
        loss = model.loss(out, graphPyg.y, weight)
        loss_train = loss.item()
        running_loss += loss_train
        running_total += 1

        # confusion matrix
        # S = train_y.data.cpu().numpy()
        S = graphPyg.y.detach().clone().cpu().numpy()
        C = np.argmax(torch.nn.Softmax(dim=0)(out).detach().clone().cpu().numpy(), axis=1)
        CM = confusion_matrix(S, C).astype(np.float32)
        nb_classes = CM.shape[0]
        train_y = graphPyg.y.detach().clone().cpu().numpy()
        for r in range(nb_classes):
            cluster = np.where(train_y == r)[0]
            CM[r, :] /= cluster.shape[0]
        running_conf_mat += CM
        running_accuracy += np.sum(np.diag(CM))/ nb_classes

        # confusion matrix
        average_conf_mat = running_conf_mat / running_total
        average_accuracy = running_accuracy / running_total
        average_loss = running_loss / running_total

    # print results
    print('\nloss(100 pre-saved data)= %.3f, accuracy(100 pre-saved data)= %.3f' % (average_loss, 100 * average_accuracy))


    #############
    # output
    #############
    result = {}
    result['final_loss'] = average_loss
    result['final_acc'] = 100 * average_accuracy
    result['final_CM'] = 100 * average_conf_mat
    result['final_batch_time'] = t_stop
    result['nb_param_nn'] = nb_param
    result['plot_all_epochs'] = tab_results
    print(result)



if __name__ == '__main__':
    main()
