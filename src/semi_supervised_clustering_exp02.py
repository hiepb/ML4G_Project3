from original_model import Graph_OurConvNet
from pyg_model import RGGConvModel
import block
import graph_generator as g
import utils

import os
import time
from datetime import timedelta
import argparse
import pathlib
import pickle
from collections import OrderedDict
import numpy as np
import random
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
# pip freeze > requirements.txt


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
    parser.add_argument('--model', type=str, default='pyg', choices=['orig', 'pyg'])
    parser.add_argument('--max_iters', type=int, default=5000)
#    parser.add_argument('--batch_iters', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.00075)
    parser.add_argument('--decay_rate', type=float, default=1.25)
    parser.add_argument('--no_cuda', help='Set this if cuda is availbable, but you do NOT want to use it.', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--numComm', type=int, default=10, help='Number of communities in the graph')
    parser.add_argument('dataDir', type=str, help='path to the data directory')
    parser.add_argument('--sizeMin',
                        default=5,
                        type=int,
                        help='Minimum number of nodes of other communities.')
    parser.add_argument('--sizeMax',
                        default=25,
                        type=int,
                        help='Maximum number of nodes of other communities.')
    parser.add_argument('-p',
                        default=0.5,
                        type=float,
                        help='Probability that two nodes of the same community are connected.')
    parser.add_argument('-q',
                        default=0.1,
                        type=float,
                        help='Probability that two nodes of the different communities are connected.')
    parser.add_argument('--dimEmb',
                        default=50,
                        type=int,
                        help='Embedding dimension.')
    parser.add_argument('--logDir', default=None, type=str, help='tensorboardXs save directory location.')  # TODO
    parser.add_argument('--numRuns', default=5, type=int, help='The number of times the experiment should be repeated. The performance metrics gets averaged over all runs.')

    return parser.parse_args()


def lossFn(weight, out, y):
    return nn.CrossEntropyLoss(weight=weight)(out, y)


def create_legacy_parameters(args):
    """ Creates a dictionary with the subgraph matching parameters for the code of the origianl author"""
    # Create subgraph that will be searched for in the larger graph
    legacyParams = dict()
    legacyParams['Voc'] = args.vocSize
    legacyParams['nb_clusters_target'] = args.numComm
    legacyParams['size_min'] = args.sizeMin
    legacyParams['size_max'] = args.sizeMax
    legacyParams['p'] = args.p
    legacyParams['q'] = args.q
    legacyParams['D'] = args.dimEmb
    legacyParams['H'] = args.dimHid
    legacyParams['L'] = args.numLayers
    legacyParams['flag_task'] = 'clustering'
    with open(args.datasetPath, 'rb') as fp:
        all_trainx = pickle.load(fp)
    legacyParams['all_trainx'] = all_trainx[:100]
    return legacyParams


def main():

    args = parse_args()
    if args.seed:
        set_seed(args.seed)
    timeAbsolute = time.time()

    args.vocSize = args.numComm + 1
    args.datasetPath = pathlib.Path(args.dataDir)
    args.datasetPath = args.datasetPath.joinpath('set_100_clustering_maps_p05_q01_size5_25_2017-10-31_10-25-00_.txt')
    # following the paper parameters
    args.dimHid = 50
    print(args)

    if args.logDir is None:
        logDir = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    else:
        logDir = args.logDir

    use_cuda = False
    if torch.cuda.is_available() and not args.no_cuda:
        use_cuda = True
        print('Using CUDA: TRUE')
    else:
        print('Using CUDA: FALSE')

    data = OrderedDict()
    for curL in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        args.numLayers = curL
        accuracy_runs = []
        time_runs_train = []
        time_runs_eval = []
        # Repeat experiment numRuns times
        for run in range(args.numRuns):
            print(f'####################Starting run: {str(run).zfill(4)}####################')
            # Define legacy parameters for the code of the original author
            legacyParams = create_legacy_parameters(args)

            # Define model
            if args.model == 'orig':
                model = Graph_OurConvNet(legacyParams)
            elif args.model == 'pyg':
                model = RGGConvModel(dimIn=args.vocSize, dimEmb=args.dimEmb, dimHid=args.dimHid, dimOut=args.numComm, numLayers=args.numLayers)
            if use_cuda:
                model = model.cuda()

            # number of network parameters
            # numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # print(model)
            # print(f'Number of parameters: {numParams}')
            # print(f'Number of residual gated graph convolutional layers: {args.numLayers}')
            # print(f'Hidden dim: {args.dimHid}')

            # optimization parameters
            learning_rate = args.learning_rate
            max_iters = args.max_iters
            batch_iters = 100  # args.batch_iters
            decay_rate = args.decay_rate

            # Optimizer
            lr = learning_rate
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            t_start = time.time()
            t_start_total = time.time()
            average_loss_old = 1e10
            running_loss = 0.0
            running_total = 0
            running_conf_mat = 0
            running_accuracy = 0
            tab_results = []

            tensorboardPath = pathlib.Path.cwd()
            tensorboardPath = tensorboardPath.joinpath('runs', logDir, f'L_{curL}', f'run{str(run).zfill(4)}')
            writer = SummaryWriter(tensorboardPath)    # TODO
            print(f'Writing results of run {str(run).zfill(4)} to: {writer.logdir}')

            for iteration in range(max_iters):
                model.train()
                graphOrg = g.graph_semi_super_clu(legacyParams)
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

                loss = lossFn(weight, out, graphPyg.y)

                loss_train = loss.item()
                running_loss += loss_train
                running_total += 1

                CM, numClasses = utils.compute_confusion_matrix(out, graphPyg.y)

                running_conf_mat += CM
                running_accuracy += np.sum(np.diag(CM)) / numClasses

                # backward, update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # learning rate, print results
                if not iteration % batch_iters:

                    # time
                    t_stop = time.time() - t_start
                    t_start = time.time()

                    # confusion matrix
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

                    writer.add_scalar('train/loss', average_loss, iteration)
                    writer.add_scalar('train/accuracy', average_accuracy, iteration)
                    writer.add_scalar('train/time', t_stop, iteration)
                    time_runs_train.append(t_stop)

                    # print results
                    if 1 == 2:
                        print('\niteration=%d, loss(%diter)=%.3f, lr=%.8f, time(%diter)=%.2f' %
                              (iteration, batch_iters, average_loss, lr, batch_iters, t_stop))
                        #print('Confusion matrix= \n', 100* average_conf_mat)
                        print('accuracy=%.3f' % (100 * average_accuracy))

            # Testing
            running_loss = 0.0
            running_total = 0
            running_conf_mat = 0
            running_accuracy = 0
            model.eval()
            with torch.no_grad():
                timeStart_eval = time.time()
                for iteration in range(100):

                    # generate one data
                    graphOrg = legacyParams['all_trainx'][iteration][1]
                    graphPyg = utils.to_pyg_data(graphOrg)

                    if use_cuda:
                        graphPyg = graphPyg.cuda()

                    if args.model == 'orig':
                        out = model(graphOrg, use_cuda)
                    elif args.model == 'pyg':
                        out = model(graphPyg.x, graphPyg.edge_index)

                    weight = utils.compute_loss_weight(graphPyg.y)

                    if use_cuda:
                        weight = weight.cuda()

                    loss = lossFn(weight, out, graphPyg.y)
                    loss_train = loss.item()
                    running_loss += loss_train
                    running_total += 1

                    CM, numClasses = utils.compute_confusion_matrix(out, graphPyg.y)

                    running_conf_mat += CM
                    running_accuracy += np.sum(np.diag(CM)) / numClasses

                    # confusion matrix
                    # average_conf_mat = running_conf_mat / running_total
                    average_accuracy = running_accuracy / running_total
                    average_loss = running_loss / running_total

                    curAcc = np.sum(np.diag(CM)) / numClasses
                    writer.add_scalar("test/loss", loss.item(), iteration)
                    writer.add_scalar("test/accuracy", (np.sum(np.diag(CM)) / numClasses), iteration)

                timeEnd_eval = time.time() - timeStart_eval

            # print results
            print('\nloss(100 pre-saved data)=%.3f, accuracy(100 pre-saved data)=%.3f\n' % (average_loss, 100 * average_accuracy))
            accuracy_runs.append(average_accuracy)
            time_runs_eval.append(timeEnd_eval)


        # print(accuracy_runs)
        # print()
        # print(time_runs_train)
        # print()
        # print(time_runs_eval)
        # print()
        # print(f'run time: {str(timedelta(seconds=(time.time() - timeAbsolute)))}\n{args}')
        # print('##################################################################')
        data[curL] = (np.mean(accuracy_runs), np.mean(time_runs_train), np.mean(time_runs_eval))

    print(data)
    utils.create_plots(data, 'L', logDir)
    print(f'run time: {str(timedelta(seconds=(time.time() - timeAbsolute)))}\n{args}')
    print('##################################################################')

if __name__ == '__main__':
    main()
