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
post_training = 50

def get_emb_size(epochs, dataset, num_clients, batch_size, lambdc, lambds, red):
    """
    Get the number of remaining embedding components for post feature selection
    """
    files = glob.glob(f'out_feature_emb_sparse_{dataset}_lambdc{lambdc}_lambds{lambds}_red{red}_seed*.txt')
    emb_sizes = []    
    for f in files:
        out = open(f, 'r')
        substring = 'embedding components'
        lines = out.readlines()
        embs_removed = [line for line in lines if substring in line]
        out.close()
        emb_sizes.append((64-float(embs_removed[0].split(' ')[1]))/64)
    return np.average(emb_sizes)

def all_seeds(prefix, selector, epochs, dataset, num_clients, batch_size, lambdc, lambds, red):
    """
    Get results over all seeds
    """
    print(dataset, selector)
    if red == 0:
        files = glob.glob(f'{prefix}_feature_L21_BS{batch_size}_NC{num_clients}_epochs{epochs}_selector{selector}_dataset{dataset}_lambdc{lambdc}_lambds{lambds}_seed*.pkl')
    #else:
    elif selector == 'baseline_red':
        files = glob.glob(f'{prefix}_feature_L21_BS{batch_size}_NC{num_clients}_epochs{epochs}_selector{selector}_dataset{dataset}_lambdc{lambdc}_lambds{lambds}_red{red}_seed*.pkl')
    else:
        print(f'{prefix}_featureSGD_L21_BS{batch_size}_NC{num_clients}_epochs{epochs}_selector{selector}_dataset{dataset}_lambdc{lambdc}_lambds{lambds}_red{red}_seed*.pkl')
        files = glob.glob(f'{prefix}_featureSGD_L21_BS{batch_size}_NC{num_clients}_epochs{epochs}_selector{selector}_dataset{dataset}_lambdc{lambdc}_lambds{lambds}_red{red}_seed*.pkl')

    # Parse accuracy files
    pickles = []    
    min_len = 10000
    for f in files:
        pkl = pickle.load(open(f, 'rb'))
        if "accs" in prefix:
            pkl = np.array(np.array(pkl)[:,0,1], dtype=np.float32)*100
        else: 
            pkl = np.array([x.cpu().numpy() for x in pkl])
        if selector == 'centralized' and len(pkl) < 100:
            continue
        elif selector == 'sparse' and len(pkl) < 251+epochs:
            continue
        elif selector == 'emb_sparse' and len(pkl) < 401+epochs:
            continue
        min_len = min(min_len, len(pkl))
        pickles.append(pkl)
    if len(pickles) == 0:
        return None
    for i in range(len(pickles)):
        pickles[i] = pickles[i][:min_len]
    pickles = np.array(pickles)
       
    # Average results and remove non-communication rounds
    avg = np.average(pickles, axis=0)
    std = np.std(pickles, axis=0)
    if selector == 'centralized' or selector == 'baseline' or selector == 'random':
        avg = avg[0:101]
        std = std[0:101]
    elif selector == 'sparse':
        avg = np.concatenate((avg[0:epochs+1], avg[151+epochs:-epochs]))
        std = np.concatenate((std[0:epochs+1], std[151+epochs:-epochs]))
    elif selector == 'emb_sparse':
        avg = np.concatenate((avg[0:epochs+1], avg[301+epochs:-epochs]))
        std = np.concatenate((std[0:epochs+1], std[301+epochs:-epochs]))

    if selector == "emb_sparse":
        return (avg, std)
    else:
        return (avg[:post_training+1], std[:post_training+1])

t = 'accs_test'

datasets = ['mimic3', 'gina', 'sylva', 'activity', 'phishing']
batches = [1000,100,1000,1000,1000]
num_clients = [4,4,4,4,3]
reds = [0.5]
selectors = ['centralized', 'sparse', 'emb_sparse']

for red in reds: 
    centralized_lambds = [0.25, 0.1, 1.0, 0.1, 0.1]
    sparse_lambds =      [31.0, 0.025, 0.1, 0.1, 0.25]
    emb_sparse_lambdsc = [32.5, 0.025, 0.1, 0.1, 0.1]
    emb_sparse_lambdss = [0.1, 0.005, 0.01, 0.25, 0.005]
    epochs_emb = [1,1,1,2,1]
    epochs_sparse = [1,1,1,3,1]

    for dataset, epoch_emb, epoch_sparse, batch, clients, cent_lambd, sparse_lambd, emb_lambdc, emb_lambds in zip(datasets, epochs_emb, epochs_sparse, batches, num_clients, centralized_lambds, sparse_lambds, emb_sparse_lambdsc, emb_sparse_lambdss):
        # Parse results
        losses0 = all_seeds(t,"baseline", 0, dataset, clients, batch, 0.0, 0.0, 0)
        if red != 0:
            lossesbr = all_seeds(t,"baseline_red", 0, dataset, clients, batch, 0.0, 0.0, red)
        lossesc = all_seeds(t,"centralized", 0, dataset, clients, batch, cent_lambd, 0.0, red)

        losses1 = all_seeds(t,"sparse", epoch_sparse, dataset, clients, batch, sparse_lambd, 0.0, red)
        losses2 = all_seeds(t,"emb_sparse", epoch_emb, dataset, clients, batch, emb_lambdc, emb_lambds, red)

        # Plot baselines 
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
        bits_per_epoch = embedding_size*num_samples[dataset]*clients*32
        x_cost = np.linspace(0, post_training*bits_per_epoch/(2**20), post_training+1)

        plt.plot(x_cost, losses0[0], label=f'VFL (Original)', color='C3')
        plt.fill_between(x_cost, losses0[0] - losses0[1], losses0[0] + losses0[1], alpha=0.3, color='C3')
        if red != 0:
            plt.plot(x_cost, lossesbr[0], label=f'VFL (Spurious)', color='C4')
            plt.fill_between(x_cost, lossesbr[0] - lossesbr[1], lossesbr[0] + lossesbr[1], alpha=0.3, color='C4')
        plt.plot(x_cost, lossesc[0], label=f'Group Lasso', color='C2')
        plt.fill_between(x_cost, lossesc[0] - lossesc[1], lossesc[0] + lossesc[1], alpha=0.3, color='C2')
        plt.plot(x_cost, losses1[0], label='Local Lasso', color='C1')
        plt.fill_between(x_cost, losses1[0] - losses1[1], losses1[0] + losses1[1], alpha=0.3, color='C1')

        # Scale LESS-VFL on X-axis based on size of embeddings during post feature selection
        emb_size = get_emb_size(epoch_emb, dataset, clients, batch, emb_lambdc, emb_lambds, red)
        bits_per_epoch_emb = emb_size*embedding_size*num_samples[dataset]*clients*32
        x_cost_pre = np.linspace(0, (epoch_emb+1)*bits_per_epoch/(2**20), epoch_emb+2)
        steps_emb = math.ceil((post_training+1)*bits_per_epoch/bits_per_epoch_emb)
        x_cost_post = np.linspace((epoch_emb+1)*bits_per_epoch/(2**20), post_training*bits_per_epoch/(2**20), steps_emb+1)
        x_cost_emb = np.concatenate((x_cost_pre, x_cost_post))
        if len(x_cost_emb) < len(losses2[0]):
            losses20 = losses2[0][:len(x_cost_emb)]
            losses21 = losses2[1][:len(x_cost_emb)]
        else:
            losses20 = losses2[0]
            losses21 = losses2[1]
            x_cost_emb = x_cost_emb[:len(losses2[0])]
            plt.xlim(0, x_cost_emb[-1])
        plt.plot(x_cost_emb, losses20, label=f'LESS-VFL (ours)', color='C0')
        plt.fill_between(x_cost_emb, losses20 - losses21, losses20 + losses21, alpha=0.3, color='C0')

        # Plot results
        plt.xlabel('Communication Cost (MB)')
        if dataset == 'phishing' or dataset == 'sylva':
            plt.ylim(45, 103)
        else:
            plt.ylim(20, 103)
        plt.ylabel('Test Accuracy (%)')

        plt.legend(prop={'size': 16}, loc='lower right')
        plt.tight_layout()
        plt.savefig(f'{t}_featurel21_{dataset}_red{red}.png')
        #plt.savefig(f'motivation_{dataset}_red{red}.png')
        plt.close()
