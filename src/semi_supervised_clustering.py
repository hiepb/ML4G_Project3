# task_parameters = {}
# task_parameters['flag_task'] = 'clustering'
# task_parameters['nb_communities'] = 10
# task_parameters['nb_clusters_target'] = task_parameters['nb_communities']
# task_parameters['Voc'] = task_parameters['nb_communities'] + 1
# task_parameters['size_min'] = 5
# task_parameters['size_max'] = 25
# task_parameters['p'] = 0.5
# task_parameters['q'] = 0.1
# file_name = 'data/set_100_clustering_maps_p05_q01_size5_25_2017-10-31_10-25-00_.txt'
# with open(file_name, 'rb') as fp:
#     all_trainx = pickle.load(fp)
# task_parameters['all_trainx'] = all_trainx[:100]


# # network parameters
# net_parameters = {}
# net_parameters['Voc'] = task_parameters['Voc']
# net_parameters['D'] = 50
# net_parameters['nb_clusters_target'] = task_parameters['nb_clusters_target']
# net_parameters['H'] = 50
# net_parameters['L'] = 10
# #print(net_parameters)


# # optimization parameters
# opt_parameters = {}
# opt_parameters['learning_rate'] = 0.00075   # ADAM
# opt_parameters['max_iters'] = 5000
# opt_parameters['batch_iters'] = 100
# if 2==1: # fast debugging
#     opt_parameters['max_iters'] = 101
#     opt_parameters['batch_iters'] = 10
# opt_parameters['decay_rate'] = 1.25
