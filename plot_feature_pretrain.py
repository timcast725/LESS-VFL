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
import tensorflow as tf

font = {'family' : 'DejaVu Sans',
#        'weight' : 'bold',
        'size'   : 20}
plt.rc('font', **font)
colors=['#6aa2fc', '#fc8181', '#a5ff9e', '#3639ff', '#ff3636', '#13ba00', '#ff62f3']

num_features = {'mimic3': 712, 'gina': 968, 'activity': 560, 'phishing': 30, 'sylva': 216}
name = {'mimic3': 'MIMIC-III', 'gina': 'Gina', 'activity': 'Activity', 'phishing': 'Phishing', 'sylva': 'Sylva'}
input_size = {'mimic3': 178, 'gina': 242, 'activity': 140, 'phishing': 10, 'sylva': 54}
num_samples = {'mimic3': 14681, 'gina': 3468*0.8, 'activity': 7352, 'phishing': 11055*0.8, 'sylva': 14395*0.8}
embedding_size = 16
post_training = 100

def all_seeds(prefix, selector, epochs, dataset, num_clients, batch_size, lambdc, lambds, redundancy, emb_sizes):
    """
    Get results over all seeds
    """
    batch_size = 1000
    if dataset == 'gina':
        batch_size = 100
    files = glob.glob(f'importances_featureSGD_L21_BS{batch_size}_NC{num_clients}_epochs{epochs}_selector{selector}_dataset{dataset}_lambdc{lambdc}_lambds{lambds}_red{redundancy}_seed*.pkl')
    files2 = glob.glob(f'{prefix}_featureSGD_L21_BS{batch_size}_NC{num_clients}_epochs{epochs}_selector{selector}_dataset{dataset}_lambdc{lambdc}_lambds{lambds}_red{redundancy}_seed*.pkl')

    # Open file with input model weights and accuracy
    sorted_inds = None
    pickles = []
    red_pers = []
    for i, f in enumerate(files):
        # Determine percentage of spurious features removed
        seed_string = f.split('_')[-1].split('.')[0]
        importance = pickle.load(open(f, 'rb'))
        importance = tf.concat(importance, 0)
        non_zero = np.where(importance > 0)[0]
        zero = np.where(importance <= 0)[0]

        X = np.arange(0,len(importance))
        start = 0
        end = 0
        redundant_features = [] 
        for client in range(num_clients):
            start += input_size[dataset]
            end += int(input_size[dataset]*(1+redundancy))
            red = np.arange(start,end)
            redundant_features = np.concatenate((redundant_features, red))
            start = end
        correct = list(set(redundant_features).intersection(zero))
        red_pers.append(len(correct)/len(redundant_features))
        
        # Get accuracy
        try:
            pkl = pickle.load(open(files2[i], 'rb'))
        except:
            continue
        pkl = np.array(np.array(pkl)[:,0,1], dtype=np.float32)*100
        if selector == 'centralized' or selector == 'baseline':
            pkl = pkl[0:101]
        elif selector == 'sparse':
            pkl = np.concatenate((pkl[0:epochs+1], pkl[150+epochs:-epochs]))
        elif selector == 'emb_sparse':
            pkl = np.concatenate((pkl[0:epochs+1], pkl[300+epochs:-epochs]))
        pickles.append(pkl)
        if selector == 'emb_sparse':
            # Determine percentage of embeddings removed
            filename = f'/hdd/home/results_table/out_feature_emb_sparse_{dataset}_lambdc{lambdc}_lambds{lambds}_red{redundancy}_{seed_string}.txt'
            out = open(filename, 'r')
            substring = 'embedding components'
            lines = out.readlines()
            embs_removed = [line for line in lines if substring in line]
            out.close()
            emb_sizes.append((64-float(embs_removed[0].split(' ')[1]))/64)

    pickles = np.array(pickles)

    # Calculate cost during pre-training and post feature selection
    if selector == 'emb_sparse':
        embedding_per = np.average(emb_sizes)
    else:
        embedding_per = 1.0 

    bits_list = []
    bits_list_pre = []
    bits_list_post = []
    for i in range(len(pickles)):
        # Pre-training cost
        min_bits_per_epoch = min(emb_sizes)*embedding_size*num_samples[dataset]*num_clients*32
        bits_per_epoch_pre = embedding_size*num_samples[dataset]*num_clients*32
        bits_per_epoch_post = embedding_per*embedding_size*num_samples[dataset]*num_clients*32
        final_epoch = 0
        bits_so_far = 0
        if epochs > 0:
            while final_epoch <= epochs:
                bits_so_far += bits_per_epoch_pre
                final_epoch += 1
        bits_so_far_pretrain = bits_so_far

        # Post feature selection cost
        cur_red = red_pers[i]
        while final_epoch < len(pickles[i]):
            if selector == 'centralized':
                cur_red = red_pers[i][final_epoch]
            if pickles[i][final_epoch] > 80 and cur_red > 0.8:
                break
            bits_so_far += bits_per_epoch_post
            final_epoch += 1
        if final_epoch < len(pickles[i]) or (pickles[i][-1] > 80 and cur_red > 0.8):
            bits_list.append(bits_so_far)
            bits_list_pre.append(bits_so_far_pretrain)
            bits_list_post.append(bits_so_far - bits_so_far_pretrain)

    # Average and return results
    avg_bits = np.average(bits_list)/(8*(2**20))
    std_bits = np.std(bits_list)/(8*(2**20))
    avg_bits_pre = np.average(bits_list_pre)/(8*(2**20))
    std_bits_pre = np.std(bits_list_pre)/(8*(2**20))
    avg_bits_post = np.average(bits_list_post)/(8*(2**20))
    std_bits_post = np.std(bits_list_post)/(8*(2**20))
    return avg_bits, std_bits, avg_bits_pre, std_bits_pre, avg_bits_post, std_bits_post, emb_sizes

t = 'importances'
datasets = ['mimic3', 'gina', 'sylva', 'activity', 'phishing']
epochs = [1,2,3,4,5]
batches = [1000,100,1000,1000,1000]
num_clients = [4,4,4,4,3]
reds = [0.5]
centralized_lambds = [0.25, 0.1, 1.0, 0.1, 0.1]
sparse_lambds =      [40.0, 0.025, 0.1, 0.1, 0.1]
emb_sparse_lambdsc = [40.0, 0.025, 0.1, 0.1, 0.1]
emb_sparse_lambdss = [0.1, 0.005, 0.01, 0.25, 0.01]
epochs_emb = [1,1,1,2,1]
epochs_sparse = [1,1,1,3,1]

for red in reds:
    for dataset, batch, clients, cent_lambd, sparse_lambd, emb_lambdc, emb_lambds in zip(datasets, batches, num_clients, centralized_lambds, sparse_lambds, emb_sparse_lambdsc, emb_sparse_lambdss):
        print(f"\midrule")
        prefix = f'\multirow{{{len(epochs)}}}{{*}}{{{name[dataset]}}} '
        for epoch in epochs:
            # Parse results
            losses = []
            stds = []
            avg_emb, std_emb, avg_emb_pre, std_emb_pre, avg_emb_post, std_emb_post, emb_sizes = all_seeds('accs_test',"emb_sparse", epoch, dataset, clients, batch, emb_lambdc, emb_lambds, red, [])
            avg_sparse, std_sparse, avg_sparse_pre, std_sparse_pre, avg_sparse_post, std_sparse_post, _ = all_seeds('accs_test',"sparse", epoch, dataset, clients, batch, sparse_lambd, 0.0, red, emb_sizes)

            # Print results in LaTeX table
            if np.isnan(avg_emb):
                emb_string = '--'
            else:
                #emb_string = f'{avg_emb_pre:.2f} $\pm$ {std_emb_pre:.2f} & {avg_emb_post:.2f} $\pm$ {std_emb_post:.2f} & {avg_emb:.2f} $\pm$ {std_emb:.2f}'
                emb_string = f'{avg_emb_pre:.2f} & {avg_emb_post:.2f} & {avg_emb:.2f}'
            if np.isnan(avg_sparse):
                sparse_string = '--'
            else:
                #sparse_string = f'{avg_sparse_pre:.2f} $\pm$ {std_sparse_pre:.2f} & {avg_sparse_post:.2f} $\pm$ {std_sparse_post:.2f} & {avg_sparse:.2f} $\pm$ {std_sparse:.2f}'
                sparse_string = f'{avg_sparse_pre:.2f} & {avg_sparse_post:.2f} & {avg_sparse:.2f}'
            print(f"{prefix}& ${epoch}$ & {sparse_string} & {emb_string} \\\\")
