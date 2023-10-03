# LESS-VFL: Communication-Efficient Feature Selection in Vertical Federated Learning

In this repo, we provide code that simulates a vertical federated learning (VFL) setting and runs LESS-VFL, a method for feature selection in VFL.
This repo builds off of the [MIMIC-III Benchmarks repo](https://github.com/YerevaNN/mimic3-benchmarks).

### Dependencies
One can install our environment with Anaconda:
```bash
conda env create -f flearning.yml 
```

### Dataset download links

[MIMIC-III dataset](https://physionet.org/content/mimiciii-demo/1.4/)

Specific details on properly preprocessing the MIMIC-III dataset are found below.

Our scripts require that the following datasets be downloaded and placed in a subfolder labeled 'data'.

[Gina Agnostic dataset](https://www.openml.org/search?type=data&sort=runs&id=1038&status=active)

[Sylva Agnostic dataset](https://www.openml.org/search?type=data&status=active&id=40992)

[Activity dataset](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)

[Phishing Website dataset](https://www.openml.org/search?type=data&sort=runs&id=4534&status=active)

### Preprocessing the MIMIC-III dataset
First, you will need to get access to the MIMIC-III .csv files: [physionet.org/content/mimiciii-demo/1.4/](https://physionet.org/content/mimiciii-demo/1.4/)

In order to preprocess the data:
    
1. The following command takes MIMIC-III CSVs, generates one directory per `SUBJECT_ID` and writes ICU stay information to `data/{SUBJECT_ID}/stays.csv`, diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`, and events to `data/{SUBJECT_ID}/events.csv`. This step might take around an hour.
```bash
python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/
```

2. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows (more information can be found in [`mimic3benchmark/scripts/more_on_validating_events.md`](mimic3benchmark/scripts/more_on_validating_events.md)).

```bash
python -m mimic3benchmark.scripts.validate_events data/root/
```

3. The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in ```{SUBJECT_ID}/episode{#}_timeseries.csv``` (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in ```{SUBJECT_ID}/episode{#}.csv```. This script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). **Outlier detection is disabled in the current version**.

```bash
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
```

4. The next command splits the whole dataset into training and testing sets. Note that the train/test split is the same of all tasks.

```bash
python -m mimic3benchmark.scripts.split_train_and_test data/root/
```
	
5. The following command will generate a task-specific dataset for "in-hospital mortality".

```bash
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
```

After the above commands are done, there will be a directory `data/{task}` for each created benchmark task.
These directories have two sub-directories: `train` and `test`.
Each of them contains bunch of ICU stays and one file with name `listfile.csv`, which lists all samples in that particular set.
Each row of `listfile.csv` has the following form: `icu_stay, period_length, label(s)`.
A row specifies a sample for which the input is the collection of ICU event of `icu_stay` that occurred in the first `period_length` hours of the stay and the target is/are `label(s)`.
In in-hospital mortality prediction task `period_length` is always 48 hours, so it is not listed in corresponding listfiles.

#### Usage
Example for running LESS-VFL with the number of parties=4, party regularization coefficient=2.0, server regularization coefficient=0.1., pre-training epochs=2, on the Gina dataset:
```bash
python -um mimic3models.in_hospital_mortality.feature_select_l21 --num_clients 4 --network mimic3models/keras_models/dense_bottom.py --dim 64 --timestep 1.0 --depth 3 --mode train --lr 0.01 --batch_size 100 --output_dir mimic3models/in_hospital_mortality --epochs 2 --local_epochs 1 --selector emb_sparse --dataset gina --lambda_client 2.0 --lambda_server 0.1 --optimizer Adam --redundancy 0.5
```

For the MIMIC-III dataset, to speed up loading data at the start of training, pickle
files of the data are saved after the first run of feature_select_l21.py with the MIMIC dataset.
Comment out lines 200-246 and uncomment lines 249-252
after running with the MIMIC dataset once to load from pickle file. 

### Running experiments from paper
Our experiments were run in parallel on a supercomputer. 
The batching scripts have been ommitted to avoid leaking private information.
If you wish to rerun the experiments,
the script "run_feature_grid.py" runs all experiments sequentially.
We do NOT recommend running the script as given, as it will takes several days to run.
Depending on your system, we recommend running the experiments in parallel in batches.

The results from all experiments are saved as pickle files in the current working directory.
A LaTeX table showing the final training accuracy and percentage of spurious features removed
can be created by running:
```bash
python plot_feature_red.py
```
A LaTeX table showing the communication cost to reach 80% training accuracy
while removed at least 80% of spurious features:
can be created by running:
```bash
python plot_feature_pretrain.py
```

From the grid search we ran, we determine the best regularization values
and pre-training epochs for each feature selection algorithm. 
Using these values, we can visualize the performance of the algorithms
in several ways.
To plot a graph showing the percentage of spurious feature removed over 150 communication epochs:
```bash
python plot_feature_bartime.py
```
To create a LaTeX table of the communication cost to reach 80% test accuracy
while removed at least 80% of spurious features:
```bash
python plot_feature_comm.py
```
To plot the test accuracy over training against the communication cost:
```bash
python plot_feature_l21.py
```

### Citation

If you are using the MIMIC-III benchmark code in your research, please cite the following publication.
```
@article{Harutyunyan2019,
  author={Harutyunyan, Hrayr and Khachatrian, Hrant and Kale, David C. and Ver Steeg, Greg and Galstyan, Aram},
  title={Multitask learning and benchmarking with clinical time series data},
  journal={Scientific Data},
  year={2019},
  volume={6},
  number={1},
  pages={96},
  issn={2052-4463},
  doi={10.1038/s41597-019-0103-9},
  url={https://doi.org/10.1038/s41597-019-0103-9}
}
```
**Please be sure also to cite the original [MIMIC-III paper](http://www.nature.com/articles/sdata201635).**
The `mimic3benchmark/scripts` directory contains scripts for creating the benchmark datasets.
The reading tools are in `mimic3benchmark/readers.py`.
All evaluation scripts are stored in the `mimic3benchmark/evaluation` directory.
The `mimic3models` directory contains the baselines models along with some helper tools.
Those tools include discretizers, normalizers and functions for computing metrics.

