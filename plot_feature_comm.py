"""
Plot adaptive experimental results
"""
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import os
import glob
import math
import scipy.interpolate as interp

font = {'family' : 'DejaVu Sans',
#        'weight' : 'bold',
        'size'   : 20}
plt.rc('font', **font)
colors=['#6aa2fc', '#fc8181', '#a5ff9e', '#3639ff', '#ff3636', '#13ba00', '#ff62f3']

input_size = {'mimic3': 178, 'gina': 242, 'activity': 140, 'phishing': 10, 'sylva': 54}
num_samples = {'mimic3': 14681, 'gina': 3468*0.8, 'activity': 7352, 'phishing': 11055*0.8, 'sylva': 14395*0.8}
embedding_size = 16
post_training = 100

def all_seeds(prefix, selector, epochs, dataset, num_clients, batch_size, lambdc, lambds, emb_sizes, red):
    """
    Averages the results of all seeds
    """
    files = glob.glob(f'{prefix}_featureSGD_L21_BS{batch_size}_NC{num_clients}_epochs{epochs}_selector{selector}_dataset{dataset}_lambdc{lambdc}_lambds{lambds}_red{red}_seed*.pkl')
    files_base = glob.glob(f'/hdd/home/feature_results/{prefix}_feature_L21_BS{batch_size}_NC{num_clients}_epochs0_selectorbaseline_dataset{dataset}_lambdc0.0_lambds0.0_red0.0_seed*.pkl')
    pickles = []    
    pickles_bas = []    
    min_len = 10000
    for f,f_bas in zip(files, files_base):
        # Get accuracy pickles
        seed = f.split('_')[-1][4:-4]
        pkl = pickle.load(open(f, 'rb'))
        pkl_bas = pickle.load(open(f_bas, 'rb'))
        pkl = np.array(np.array(pkl)[:,0,1], dtype=np.float32)
        pkl_bas = np.array(np.array(pkl_bas)[:,0,1], dtype=np.float32)*100
        pickles_bas.append(pkl_bas)
        if selector == 'sparse':
            pkl = np.concatenate((pkl[0:epochs+1], pkl[150+epochs:]))
        elif selector == 'emb_sparse':
            pkl = np.concatenate((pkl[0:epochs+1], pkl[300+epochs:]))
        pickles.append(pkl*100)

        if selector == 'emb_sparse':
            # Get percentage of embedding components kept during post feature selection
            filename = f'out_feature_emb_sparse_{dataset}_lambdc{lambdc}_lambds{lambds}_red{red}_seed{seed}.txt'
            out = open(filename, 'r')
            substring = 'embedding components'
            lines = out.readlines()
            embs_removed = [line for line in lines if substring in line]
            out.close()
            emb_sizes.append((64-float(embs_removed[0].split(' ')[1]))/64)

    target = np.max(pickles_bas)
    pickles = np.array(pickles)

    if selector == 'centralized':
        # Get percentage of spurious features removed for all iterations
        files_imp = glob.glob(f'importances_all_featureSGD_L21_BS{batch_size}_NC{num_clients}_epochs{epochs}_selector{selector}_dataset{dataset}_lambdc{lambdc}_lambds{lambds}_red{red}_seed*.pkl')
        red_pers = []
        for i, f in enumerate(files_imp):
            importances = pickle.load(open(f, 'rb'))[:-1]
            red_per = []
            for importance in importances:
                correct = 0
                total = 0
                for client in range(num_clients):
                    start = input_size[dataset]
                    end = int(input_size[dataset]*(1+red))
                    red_new = np.arange(start,end)
                    total += len(red_new)
                    correct += len(list(set(red_new).intersection(importance[client])))
                red_per.append(correct/total)
            while len(red_per) < len(pickles[i]):
                red_per.append(red_per[-1])
            red_pers.append(red_per)
    else:
        # Get percentage of spurious features removed total 
        files = glob.glob(f'importances_featureSGD_L21_BS{batch_size}_NC{num_clients}_epochs{epochs}_selector{selector}_dataset{dataset}_lambdc{lambdc}_lambds{lambds}_red{red}_seed*.pkl')
        red_pers = []
        for i, f in enumerate(files):
            importance = pickle.load(open(f, 'rb'))
            importance = tf.concat(importance, 0)
            zero = np.where(importance <= 0)[0]
            start = 0
            end = 0
            redundant_features = [] 
            for client in range(num_clients):
                start += input_size[dataset]
                end += int(input_size[dataset]*(1+red))
                redundancy = np.arange(start,end)
                redundant_features = np.concatenate((redundant_features, redundancy))
                start = end
            correct = list(set(redundant_features).intersection(zero))
            red_pers.append(len(correct)/len(redundant_features))

    # Get size of embeddings
    if selector == 'emb_sparse':
        embedding_per = np.average(emb_sizes)
    else:
        embedding_per = 1.0 

    # Calculate the bits to reach targets
    bits_list = []
    for i in range(len(pickles)):
        if i >= len(red_pers):
            continue
        min_bits_per_epoch = min(emb_sizes)*embedding_size*num_samples[dataset]*num_clients*32
        bits_per_epoch_pre = embedding_size*num_samples[dataset]*num_clients*32
        bits_per_epoch_post = embedding_per*embedding_size*num_samples[dataset]*num_clients*32

        # Cost during pre-training
        final_epoch = 0
        bits_so_far = 0
        if epochs > 0:
            while final_epoch <= epochs:
                bits_so_far += bits_per_epoch_pre
                final_epoch += 1

        # Cost after pre-training
        cur_red = red_pers[i] 
        while final_epoch < len(pickles[i]):
            if selector == 'centralized':
                cur_red = red_pers[i][final_epoch]
            if pickles[i][final_epoch] > target*0.9 and cur_red > 0.8:
                break
            bits_so_far += bits_per_epoch_post
            final_epoch += 1
        if final_epoch < len(pickles[i]) or (pickles[i][-1] > target*0.9 and cur_red > 0.8):
            bits_list.append(bits_so_far)

    # Average and print results
    avg_bits = np.average(bits_list)
    std_bits = np.std(bits_list)
    print(f'{avg_bits/(8*(2**20)):.2f} $\pm$ {std_bits/(8*(2**20)):.2f}')

    return emb_sizes

types = ['accs_test']
datasets = ['mimic3', 'gina', 'sylva', 'activity', 'phishing']
batches = [1000,100,1000,1000,1000]
num_clients = [4,4,4,4,3]

centralized_lambds = [0.25, 0.1, 1.0, 0.1, 0.25]
sparse_lambds =      [40.0, 0.025, 0.1, 0.1, 0.1]
emb_sparse_lambdsc = [40.0, 0.025, 0.1, 0.1, 0.1]
emb_sparse_lambdss = [0.1, 0.005, 0.01, 0.25, 0.005]
epochs_emb = [1,1,1,2,1]
epochs_sparse = [1,1,1,3,1]

for t in types:
    for dataset, epoch_emb, epoch_sparse, batch, clients, cent_lambd, sparse_lambd, emb_lambdc, emb_lambds in zip(datasets, epochs_emb, epochs_sparse, batches, num_clients, centralized_lambds, sparse_lambds, emb_sparse_lambdsc, emb_sparse_lambdss):
        # Print communication cost to reach target accuracy and spurious features removed
        emb_sizes = all_seeds(t,"emb_sparse", epoch_emb, dataset, clients, batch, emb_lambdc, emb_lambds, [], 0.5)
        _ = all_seeds(t,"centralized", 0, dataset, clients, batch, cent_lambd, 0.0, emb_sizes, 0.5)
        _ = all_seeds(t,"sparse", epoch_sparse, dataset, clients, batch, sparse_lambd, 0.0, emb_sizes, 0.5)

        print('\midrule')
        print()
