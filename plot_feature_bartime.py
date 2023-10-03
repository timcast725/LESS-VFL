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
post_training = 150

def all_seeds(prefix, selector, epochs, dataset, num_clients, batch_size, lambdc, lambds, redundancy):
    """
    Averages the results of all seeds
    """
    print(dataset, selector, redundancy, lambdc, lambds)
    if selector == 'centralized':
        files = glob.glob(f'importances_all_featureSGD_L21_BS{batch_size}_NC{num_clients}_epochs{epochs}_selector{selector}_dataset{dataset}_lambdc{lambdc}_lambds{lambds}_red{redundancy}_seed*.pkl')
    else:
        files = glob.glob(f'{prefix}_featureSGD_L21_BS{batch_size}_NC{num_clients}_epochs{epochs}_selector{selector}_dataset{dataset}_lambdc{lambdc}_lambds{lambds}_red{redundancy}_seed*.pkl')

    sorted_inds = None
    red_pers = []
    # For all seeds
    for i, f in enumerate(files):
        if selector == 'centralized':
            # For group lasso:
            # get percentage of spurious features removed at each iteration
            importances = pickle.load(open(f, 'rb'))
            red_per = []
            for k, importance in enumerate(importances):
                if k >= post_training:
                    break
                correct = 0
                total = 0
                for client in range(num_clients):
                    start = input_size[dataset]
                    end = int(input_size[dataset]*(1+redundancy))
                    red_new = np.arange(start,end)
                    total += len(red_new)
                    correct += len(list(set(red_new).intersection(importance[client])))
                red_per.append(100*correct/total)
            red_pers.append(red_per)
        else:
            # For LESS-VFL and Local Lasso:
            # Get final number of spurious features removed
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
            red_per = 100*len(correct)/len(redundant_features)

            # Make list that is zero for pre-training,
            # and the value of features removed for all communication epochs after
            red_per_list = np.concatenate(([0]*epochs, [red_per]*(post_training-epochs)))
            red_pers.append(red_per_list)
        
    # Average and return results
    avg_red = np.average(red_pers, axis=0)
    std_red = np.std(red_pers, axis=0)
    print(avg_red)
    return avg_red, std_red 

t = 'importances'

datasets = ['mimic3', 'gina', 'sylva', 'activity', 'phishing']
Datasets = ['MIMIC-III','Gina', 'Sylva', 'Activity', 'Phishing']
epochs = [1,1,4,2,1]
batches = [1000,100,1000,1000,1000]
num_clients = [4,4,4,4,3]
reds = [0.5]

centralized_lambds = [0.25, 0.1, 1.0, 0.1, 0.1]
sparse_lambds =      [31.0, 0.025, 0.1, 0.1, 0.25]
emb_sparse_lambdsc = [32.5, 0.025, 0.1, 0.1, 0.25]
emb_sparse_lambdss = [0.05, 0.005, 0.01, 0.1, 0.01]
epochs_emb = [1,1,1,1,1]
epochs_sparse = [1,1,1,3,1]

for red in reds:
    for dataset, epoch_emb, epoch_sparse, batch, clients, cent_lambd, sparse_lambd, emb_lambdc, emb_lambds in zip(datasets, epochs_emb, epochs_sparse, batches, num_clients, centralized_lambds, sparse_lambds, emb_sparse_lambdsc, emb_sparse_lambdss):
        if dataset == 'activity' or dataset == 'phishing':
            font = {'family' : 'DejaVu Sans',
                    'size'   : 20}
            plt.rc('font', **font)
            fig, ax = plt.subplots(1, 1, figsize=(7, 5), facecolor='white')
        else:
            font = {'family' : 'DejaVu Sans',
                    'size'   : 20}
            plt.rc('font', **font)
            fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')

        # Get percentage of spurious features removed average and standard deviation
        avg_red_cent, std_red_cent = all_seeds(t,"centralized", 0, dataset, clients, batch, cent_lambd, 0.0, red)
        avg_red_sparse, std_red_sparse = all_seeds(t,"sparse", epoch_sparse, dataset, clients, batch, sparse_lambd, 0.0, red)
        avg_red_emb, std_red_emb = all_seeds(t,"emb_sparse", epoch_emb, dataset, clients, batch, emb_lambdc, emb_lambds, red)

        # Set position on X axis
        br1 = np.arange(post_training)
        
        # Make the plot
        plt.plot(br1+1, avg_red_cent, color ='C2', label ='Group Lasso', linewidth=5, linestyle='dashdot')
        plt.plot(br1+1, avg_red_sparse, color ='C1', label ='Local Lasso', linewidth=5)
        plt.plot(br1+1, avg_red_emb, color ='C0', label ='LESS-VFL (ours)', linewidth=5, linestyle='dashed')

        plt.ylabel('Spurious Features Removed (%)', prop={'size': 18})
        plt.xlabel('Epochs of Communication')
        plt.ylim(-3,103)
        if dataset == 'phishing':
            plt.legend(prop={'size': 18}, loc=(0.1, 0.05))
        elif dataset == 'activity':
            plt.legend(prop={'size': 18}, loc='lower right')
        else:
            plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'{t}_featurebar_{dataset}.png')
        plt.close()
