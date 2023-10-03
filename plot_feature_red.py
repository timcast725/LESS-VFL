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

def all_seeds(prefix, selector, epochs, dataset, num_clients, batch_size, lambdc, lambds, redundancy):
    """
    Get results over all seeds
    """
    batch_size = 1000
    if dataset == 'gina':
        batch_size = 100
    files = glob.glob(f'{prefix}_featureSGD_L21_BS{batch_size}_NC{num_clients}_epochs{epochs}_selector{selector}_dataset{dataset}_lambdc{lambdc}_lambds{lambds}_red{redundancy}_seed*.pkl')
    files2 = glob.glob(f'accs_train_featureSGD_L21_BS{batch_size}_NC{num_clients}_epochs{epochs}_selector{selector}_dataset{dataset}_lambdc{lambdc}_lambds{lambds}_red{redundancy}_seed*.pkl')

    sorted_inds = None
    accs = []
    red_pers = []
    for i, f in enumerate(files):
        # Determine spurious features removed
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
        
        # Open accuracy file
        try:
            pkl = pickle.load(open(files2[i], 'rb'))
        except:
            continue
        if selector == 'centralized' and len(pkl) < 100:
            continue
        elif selector == 'sparse' and len(pkl) < 251+epochs:
            continue
        elif selector == 'emb_sparse' and len(pkl) < 401+epochs:
            continue
        try:
            pkl = pkl[:pkl.index([0])]
        except:
            pass
        pkl = np.array(np.array(pkl)[:,0,1], dtype=np.float32)*100
        red_pers.append(len(correct)/len(redundant_features)*100)
        accs.append(pkl[-50])

    # If not files open, retry and don't remove results from runs that were cut short 
    if len(accs) == 0:
        for i, f in enumerate(files):
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

            try:
                pkl = pickle.load(open(files2[i], 'rb'))
            except:
                continue
            try:
                pkl = pkl[:pkl.index([0])]
            except:
                pass
            pkl = np.array(np.array(pkl)[:,0,1], dtype=np.float32)*100
            red_pers.append(len(correct)/len(redundant_features)*100)
            accs.append(pkl[-1])

    # Average and return results
    avg_acc = np.average(accs)
    std_acc = np.std(accs)
    avg_red = np.average(red_pers)
    std_red = np.std(red_pers)
    return avg_acc, std_acc, avg_red, std_red 

t = 'importances'

datasets = ['mimic3', 'gina', 'sylva', 'activity', 'phishing']
epochs_emb = [1,1,4,1,1]
epochs_sparse = [1,1,4,1,1]
batches = [1000,100,1000,1000,1000]
num_clients = [4,4,4,4,3]
reds = [0.5]
emb_sparse_lambdss = [0.25, 0.1, 0.1, 0.25, 0.25]

for red in reds:
    for dataset, epoch_emb, epoch_sparse, batch, clients, lambds in zip(datasets, epochs_emb, epochs_sparse, batches, num_clients, emb_sparse_lambdss):
        # Set all regularization values to print 
        if dataset == "mimic3":
            lambdcs = [40.0, 35.0, 32.5, 30.0, 25.0, 20.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
        elif dataset == "gina":
            lambdcs = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
        else:
            lambdcs = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
        print(f"\midrule")
        prefix = f'\multirow{{{len(lambdcs)}}}{{*}}{{{name[dataset]}}} '
        for lambdc in lambdcs:
            if dataset == "sylva" or dataset == "phishing":
                lambdss = [0.01, 0.005]
            elif dataset == "gina":
                lambdss = [0.1, 0.05, 0.01, 0.005]
            else:
                lambdss = [0.5, 0.25, 0.1, 0.05]
            for lambds in lambdss:
                # Parse results
                avg_acc_cent, std_acc_cent, avg_red_cent, std_red_cent = all_seeds(t,"centralized", 0, dataset, clients, batch, lambdc, 0.0, red)
                avg_acc_sparse, std_acc_sparse, avg_red_sparse, std_red_sparse = all_seeds(t,"sparse", epoch_sparse, dataset, clients, batch, lambdc, 0.0, red)
                avg_acc_emb, std_acc_emb, avg_red_emb, std_red_emb = all_seeds(t,"emb_sparse", epoch_emb, dataset, clients, batch, lambdc, lambds, red)

                # Print results in LaTeX table
                if np.isnan(avg_acc_cent):
                    cent_string = '-- & --'
                else:
                    cent_string = f'{avg_acc_cent:.2f} $\pm$ {std_acc_cent:.2f} & {avg_red_cent:.2f} $\pm$ {std_red_cent:.2f}'
                if np.isnan(avg_acc_sparse):
                    sparse_string = '-- & --'
                else:
                    sparse_string = f'{avg_acc_sparse:.2f} $\pm$ {std_acc_sparse:.2f} & {avg_red_sparse:.2f} $\pm$ {std_red_sparse:.2f}'
                if np.isnan(avg_acc_emb):
                    emb_string = '-- & --'
                else:
                    emb_string = f'{avg_acc_emb:.2f} $\pm$ {std_acc_emb:.2f} & {avg_red_emb:.2f} $\pm$ {std_red_emb:.2f}'
                print(f"{prefix}& $({lambdc},{lambds})$ & {cent_string} & {sparse_string} & {emb_string} \\\\")
                prefix = '                      '

