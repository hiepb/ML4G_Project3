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
import torch.nn as nn
from tqdm import tqdm

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
file_name = '/home/florian/GraphML/ResGraph/ML4G_Project3/data/set_100_subgraphs_p05_size20_Voc3_2017-10-31_10-23-00_.txt'
with open(file_name, 'rb') as fp:
    all_trainx = pickle.load(fp)
task_parameters['all_trainx'] = all_trainx[:100]



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
    parser.add_argument('--model', type=str, default='orig', choices=['orig', 'pyg'])
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--batch_iters', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.00075)
    parser.add_argument('--decay_rate', type=float, default=1.25)
    parser.add_argument('--no_cuda', help='Set this if cuda is availbable, but you do NOT want to use it.', action='store_true')
    parser.add_argument('--seed', type=str, default=42)
    parser.add_argument('--sizeSubgraph',
                        default=20,
                        type=int,
                        help='The number of nodes in the subgraph.')
    parser.add_argument('--vocSize',
                        default=3,
                        type=int,
                        help='The signal on the subgraph is generated with a uniform random distribution with a vocabulary of size vocSize.')
    parser.add_argument('--sizeMin',
                        default=15,
                        type=int,
                        help='Minimum number of nodes of other communities.')
    parser.add_argument('--sizeMax',
                        default=15,
                        type=int,
                        help='Maximum number of nodes of other communities.')
    parser.add_argument('-p',
                        default=0.5,
                        type=float,
                        help='Probability that two nodes of the same community are connected.')
    parser.add_argument('-q',
                        default=0.5,
                        type=float,
                        help='Probability that two nodes of the different communities are connected.')
    parser.add_argument('--dimEmb',
                        default=50,
                        type=int,
                        help='Embedding dimension.')
    parser.add_argument('--dimHid',
                        default=50,
                        type=int,
                        help='Hidden dimension.')
    parser.add_argument('--numLayers',
                        default=10,
                        type=int,
                        help='Number of residual gated graph convolutional layers.')
    parser.add_argument('--logging',default=True,type=bool,help='Activate/deactivate logging on tensorboard')

    return parser.parse_args()


def create_subgraph(sizeSubgraph=20, p=0.5, vocSize=3):
    """Creates the subgraph we will be looking for through a stochstic block
    model (SBM):
    SBM is a random graph which assigns communities to each node as follows:
    :param int sizeSubgraph: The number of nodes in the subgraph
    :param float p: Any two vertices in the subgraph are connected with the probability p if they belong to the same community
    :param float q: Any two vertices in the subgraph are connected with the probability q if they belong to different communities
    :param int vocSize: The signal on the subgraph is generated with a uniform random distribution with a vocabulary of size vocSize

    """
    adjMatrix = block.random_graph(sizeSubgraph, p)
    nodeFeatures = np.random.randint(vocSize, size=sizeSubgraph)
    return adjMatrix, nodeFeatures

def lossFn(weight, out, y):
    return nn.CrossEntropyLoss(weight=weight)(out, y)


def main():
    args = parse_args()
    set_seed(args.seed)
    print(args)
    use_cuda = False
    if torch.cuda.is_available() and not args.no_cuda:
        use_cuda = True
        print('Using CUDA: TRUE')
    else:
        print('Using CUDA: FALSE')

    if args.logging:
        writer = utils.TensorBoard()

    subgraph_adjMatrix, subgraph_features = create_subgraph(args.sizeSubgraph, args.p, args.vocSize)

    # garph params for original code
    graphParams = dict()
    graphParams['Voc'] = args.vocSize
    graphParams['nb_clusters_target'] = 2
    graphParams['size_min'] = args.sizeMin
    graphParams['size_max'] = args.sizeMax
    graphParams['p'] = args.p
    graphParams['q'] = args.q
    graphParams['W0'] = subgraph_adjMatrix
    graphParams['u0'] = subgraph_features

    if args.model == 'orig':
        # network parameters for original code
        net_parameters = {}
        net_parameters['Voc'] = args.vocSize
        net_parameters['D'] = args.dimEmb
        net_parameters['nb_clusters_target'] = 2
        net_parameters['H'] = args.dimHid
        net_parameters['L'] = args.numLayers
        # custom
        net_parameters['flag_task'] = 'matching'

        model = Graph_OurConvNet(net_parameters)
    elif args.model == 'pyg':
        model = RGGConvModel(dimIn=args.vocSize, dimEmb=args.dimEmb, dimHid=args.dimHid, dimOut=2, numLayers=args.numLayers)

    if use_cuda:
        model = model.cuda()

    print(model)

    # number of network parameters
    numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f'Number of parameters: {numParams}')
    print(f'Number of residual gated graph convolutional layers: {args.numLayers}')

    # optimization parameters
    learning_rate = args.learning_rate
    max_iters = args.max_iters
    batch_iters = args.batch_iters
    decay_rate = args.decay_rate

    # Optimizer
    global_lr = learning_rate
    global_step = 0
    lr = learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = model.update(lr)

    t_start = time.time()
    t_start_total = time.time()
    average_loss_old = 1e10
    running_loss = 0.0
    running_total = 0
    running_conf_mat = 0
    running_accuracy = 0
    tab_results = []

    for iteration in range(max_iters):
        graphOrg = g.variable_size_graph(graphParams)
        graphPyg = utils.to_pyg_data(graphOrg)
        # print(graphPyg)
        # print(graphPyg.y)
        if use_cuda:
            graphPyg = graphPyg.cuda()

        if args.model == 'orig':
            out = model(graphOrg, use_cuda)
        elif args.model == 'pyg':
            out = model(graphPyg.x, graphPyg.edge_index)

        weight = utils.compute_loss_weight(graphPyg.y)
        if use_cuda:
            weight = weight.cuda()

        # loss = nn.CrossEntropyLoss(weight=weight)(out, graphPyg.y)
        loss = lossFn(weight, out, graphPyg.y)

        loss_train = loss.item()
        running_loss += loss_train
        running_total += 1

        CM, numClasses = utils.compute_confusion_matrix(out, graphPyg.y)

        running_conf_mat += CM
        running_accuracy += np.sum(np.diag(CM)) / numClasses

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
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            running_loss = 0.0
            running_total = 0

            # save intermediate results
            tab_results.append([iteration, average_loss, 100 * average_accuracy, time.time()-t_start_total])

            if(args.logging == True):
                writer.loss_step_summarywriter(iteration,average_loss)
                writer.acc_step_summarywriter(iteration,100 * average_accuracy)

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
        if args.model == 'orig':
            out = model(graphOrg, use_cuda)
        elif args.model == 'pyg':
            out = model(graphPyg.x, graphPyg.edge_index)
        # y = net.forward(train_x)
        # compute loss weigth
        # labels = train_y.data.cpu().numpy()

        weight = utils.compute_loss_weight(graphPyg.y)

        if use_cuda:
            weight = weight.cuda()

        loss = lossFn(weight, out, graphPyg.y)
        loss_train = loss.item()
        running_loss += loss_train
        running_total += 1

        CM, numClasses = utils.compute_confusion_matrix(out, graphPyg.y)

        running_conf_mat += CM
        running_accuracy += np.sum(np.diag(CM))/ numClasses

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
    result['nb_param_nn'] = numParams
    result['plot_all_epochs'] = tab_results
    print(result)



if __name__ == '__main__':
    main()
