import numpy as np
import random 
import os

seeds = [1234420321,1678517510,1679295649,2141512896,755346466]
lambdss = [0.5, 0.25, 0.1, 0.05, 0.01, 0.005]
lambdcs = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
lambdcs2 = [40.0, 35.0, 32.5, 30.0, 25.0, 20.0]
datasets = ['mimic3', 'gina', 'sylva', 'activity', 'phishing']
batches = [1000,100,1000,1000,1000]
num_clients = [4,4,4,4,3]
red = 0.5
pretrain_epochs = [1,2,3,4,5]

for seed in seeds:
    for dataset, batch, clients in zip(datasets, batches, num_clients):
        os.system(f'./run_feature_slurmred.sh {seed} baseline 0 0 {dataset} 0 {batch} {clients} 0 > out_feature_baseline_{dataset}_red0_seed{seed}.txt')
        os.system(f'./run_feature_slurmred.sh {seed} baseline_red 0 0 {dataset} 0 {batch} {clients} {red} > out_feature_baselinered_{dataset}_red{red}_seed{seed}.txt')
        if dataset == 'mimic3':
            for epoch in pretrain_epochs:
                for lambdc in lambdcs2:
                    os.system(f'./run_feature_slurmred.sh {seed} sparse {lambdc} 0 {dataset} {epoch} {batch} {clients} {red} > out_feature_sparse_{dataset}_lambdc{lambdc}_lambds0_red{red}_seed{seed}.txt')
                    for lambds in lambdss:
                        os.system(f'./run_feature_slurmred.sh {seed} emb_sparse {lambdc} {lambds} {dataset} {epoch} {batch} {clients} {red} > out_feature_emb_sparse_{dataset}_epoch{epoch}_{lambdc}_{lambds}_red{red}_seed{seed}.txt')
        for lambdc in lambdcs:
            os.system(f'./run_feature_slurmred.sh {seed} centralized {lambdc} 0 {dataset} 0 {batch} {clients} {red} > out_feature_centralized_{dataset}_lambdc{lambdc}_lambds0_red{red}_seed{seed}.txt')
            if dataset != "mimic3":
                for epoch in pretrain_epochs:
                    os.system(f'./run_feature_slurmred.sh {seed} sparse {lambdc} 0 {dataset} {epoch} {batch} {clients} {red} > out_feature_sparse_{dataset}_lambdc{lambdc}_lambds0_red{red}_seed{seed}.txt')
                    for lambds in lambdss:
                        os.system(f'./run_feature_slurmred.sh {seed} emb_sparse {lambdc} {lambds} {dataset} {epoch} {batch} {clients} {red} > out_feature_emb_sparse_{dataset}_epoch{epoch}_{lambdc}_{lambds}_red{red}_seed{seed}.txt')
