"""
"""

import numpy as np
import argparse
import os
import imp
import re
import math
import copy
import random
import itertools
import time

from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer, StandardScaler
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader, LengthOfStayReader, PhenotypingReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.eager import backprop
import pickle
from sklearn import metrics as skmetrics
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras import regularizers
from keras.layers import Input
from tensorflow.keras import backend as K

import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS

input_size = {'mimic3': 178, 'gina': 242, 'activity': 140, 'phishing': 10, 'sylva': 54, 'symptoms': 7, 'ailerons': 10}

threshold = 1e-8
threshold_client = 1e-8

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def read_and_extract_features(reader, period, features):
    """
    Extract features for MIMIC-III dataset
    """
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])

class SparsifyLoss(tf.keras.losses.Loss):
    """
    Custom loss for getting mean-squared error between two embeddings
    """
    def __init__(self, lambda_weight=0.01):
        super().__init__()
        self.lambda_weight = lambda_weight
    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_pred-y_true))
        # weight_penalty = self.lambda_weight*tf.reduce_mean(tf.square(weights))
        return mse # + weight_penalty

def argparser():
    """
    Parse input arguments
    """
    import sys
    workers = int(sys.argv[2])
    parser = argparse.ArgumentParser()
    common_utils.add_common_arguments(parser)
    parser.add_argument('--seed', type=int, nargs='?', default=42,
                            help='Random seed to be used.')
    parser.add_argument('--target_repl_coef', type=float, default=0.0)
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--num_clients', type=int, help='Number of clients to split data between vertically',
                        default=2)
    parser.add_argument('--local_epochs', type=int, help='Number of local epochs to run at each client before synchronizing',
                        default=1)
    parser.add_argument('--quant_level', type=int, help='Level of quantization on embeddings',
                        default=0)
    parser.add_argument('--correct', type=bool, help='Add error correction to algorithm',
                        default=False)
    parser.add_argument('--vecdim', type=int, help='Vector quantization dimension',
                        default=1)
    parser.add_argument('--comp', type=str, help='Which compressor', default="")
    parser.add_argument('--task', type=str, help='Classification or regression task', default="classification")
    parser.add_argument('--selector', type=str, help='How to select features', default="emb_sparse")
    parser.add_argument('--dataset', type=str, help='Which dataset to use', default="mimic3")
    parser.add_argument('--embedding_size', type=int, help='Number of embedding components', default=16)
    parser.add_argument('--lambda_server', type=float, help='Regularizer coefficient on server model for sparsification', default=0.01)
    parser.add_argument('--lambda_client', type=float, help='Regularizer coefficient on client model for sparsification', default=0.01)
    parser.add_argument('--feature_per', type=float, help='Percent of features to remove (only used when --selector random)', default=1.0)
    parser.add_argument('--redundancy', type=float, help='Number of gaussian noise features to add as percentage of existing features', default=0.0)
                        
    args = parser.parse_args()
    print("*"*80, "\n\n", args, "\n\n", "*"*80)
    return args

if __name__ == "__main__":
    # Parse input arguments
    args = argparser()
    suffix = f'_featureSGD_L21_BS{args.batch_size}_NC{args.num_clients}_epochs{args.epochs}_selector{args.selector}_dataset{args.dataset}_lambdc{args.lambda_client}_lambds{args.lambda_server}_red{args.redundancy}_seed{args.seed}'
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)
    num_clients = args.num_clients
    local_epochs = args.local_epochs
    lr = args.lr

    num_classes = 1
    if args.dataset == 'mimic3':
        """
        Uncomment first block if first time running.
        Use second block after running once for faster startup.
        """
        # Build readers, discretizers, normalizers
        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                                listfile=os.path.join(args.data, 'train_listfile.csv'),
                                                period_length=48.0)

        val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'val_listfile.csv'),
                                            period_length=48.0)

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                                listfile=os.path.join(args.data, 'test_listfile.csv'),
                                                period_length=48.0)

        print('Reading data and extracting features ...')
        (train_X, train_y, train_names) = read_and_extract_features(train_reader, args.period, args.features)
        (val_X, val_y, val_names) = read_and_extract_features(val_reader, args.period, args.features)
        (test_X, test_y, test_names) = read_and_extract_features(test_reader, args.period, args.features)
        print('  train data shape = {}'.format(train_X.shape))
        print('  validation data shape = {}'.format(val_X.shape))
        print('  test data shape = {}'.format(test_X.shape))

        print('Imputing missing values ...')
        imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy=True)
        imputer.fit(train_X)
        train_X = np.array(imputer.transform(train_X), dtype=np.float32)
        val_X = np.array(imputer.transform(val_X), dtype=np.float32)
        test_X = np.array(imputer.transform(test_X), dtype=np.float32)

        print('Normalizing the data to have zero mean and unit variance ...')
        scaler = StandardScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        val_X = scaler.transform(val_X)
        test_X = scaler.transform(test_X)

        train_raw = [train_X, train_y]
        val_raw = [val_X, val_y]
        test_raw = [test_X, test_y]

        # Read data from file and save to pickle
        pickle.dump(train_raw, open('train_raw_flat.pkl', 'wb'))
        pickle.dump(val_raw, open('val_raw_flat.pkl', 'wb'))
        pickle.dump(test_raw, open('test_raw_flat.pkl', 'wb'))
        
        ''' Uncomment if data has been saved to pkl file previously '''
        ## Read data from pickle
        #train_raw = pickle.load(open('train_raw_flat.pkl', 'rb'))
        #val_raw = pickle.load(open('val_raw_flat.pkl', 'rb'))
        #test_raw = pickle.load(open('test_raw_flat.pkl', 'rb'))    

        train_raw[0] = train_raw[0][:,:712]
        test_raw[0] = test_raw[0][:,:712]
        val_raw[0] = val_raw[0][:,:712]
        coords_per = int(712/num_clients)
    elif args.dataset == 'gina':
        # Read data from csv
        data = pd.read_csv('data/gina_agnostic.csv', sep=',')
        # Normalize
        labels = data.iloc[:,-1].unique()
        label_nums = np.arange(0, len(labels), 1, dtype=int)
        replace_dict = {labels[i]: label_nums[i] for i in range(len(labels))}
        data.iloc[:,-1] = data.iloc[:,-1].replace(replace_dict)
        data = data.to_numpy().astype('float')
        data[:,:-1] = (data[:,:-1] - data[:,:-1].min())/(data[:,:-1].max() - data[:,:-1].min())

        # Create train/test split
        split_idx = int(len(data)*0.8)
        train_raw = (data[:split_idx, :968], data[:split_idx, -1])
        test_raw = (data[split_idx:, :968], data[split_idx:, -1])
        coords_per = int(968/num_clients)
        
    elif args.dataset == 'phishing':
        # Read data from csv
        data = pd.read_csv('data/phishing_websites.csv', sep=',')
        # Normalize
        labels = data.iloc[:,-1].unique()
        label_nums = np.arange(0, len(labels), 1, dtype=int)
        replace_dict = {labels[i]: label_nums[i] for i in range(len(labels))}
        data.iloc[:,-1] = data.iloc[:,-1].replace(replace_dict)
        data = data.to_numpy().astype('float')
        data[:,:-1] = (data[:,:-1] - data[:,:-1].min())/(data[:,:-1].max() - data[:,:-1].min())

        # Create train/test split
        split_idx = int(len(data)*0.8)
        train_raw = (data[:split_idx, :30], data[:split_idx, -1])
        test_raw = (data[split_idx:, :30], data[split_idx:, -1])
        coords_per = int(30/num_clients)
    elif args.dataset == 'sylva':
        # Read data from csv
        data = pd.read_csv('data/sylva_agnostic.csv', sep=',')
        # Normalize
        labels = data.iloc[:,-1].unique()
        label_nums = np.arange(0, len(labels), 1, dtype=int)
        replace_dict = {labels[i]: label_nums[i] for i in range(len(labels))}
        data.iloc[:,-1] = data.iloc[:,-1].replace(replace_dict)
        data = data.to_numpy().astype('float')
        data[:,:-1] = (data[:,:-1] - data[:,:-1].min())/(data[:,:-1].max() - data[:,:-1].min())

        # Create train/test split
        split_idx = int(len(data)*0.8)
        train_raw = (data[:split_idx, :216], data[:split_idx, -1])
        test_raw = (data[split_idx:, :216], data[split_idx:, -1])
        coords_per = int(216/num_clients)

    elif args.dataset == 'activity':
        # Read data from csv
        train_data = pd.read_csv('data/activity_train.csv', sep=',')
        test_data = pd.read_csv('data/activity_test.csv', sep=',')
        # Normalize
        labels = train_data.iloc[:,-1].unique()
        num_classes = len(labels)
        label_nums = np.arange(0, len(labels), 1, dtype=int)
        replace_dict = {labels[i]: label_nums[i] for i in range(len(labels))}
        train_data.iloc[:,-1] = train_data.iloc[:,-1].replace(replace_dict)
        train_data = train_data.to_numpy().astype('float')
        test_data.iloc[:,-1] = test_data.iloc[:,-1].replace(replace_dict)
        test_data = test_data.to_numpy().astype('float')

        # Create train/test split
        train_raw = (train_data[:, :560], tf.keras.utils.to_categorical(train_data[:, -1]))
        test_raw = (test_data[:, :560], tf.keras.utils.to_categorical(test_data[:, -1]))
        coords_per = int(560/num_clients)
        
    else:
        print("Unsupported dataset:", args.dataset)
        exit(1)

    # Add redundant features
    if args.redundancy > 0.0 and args.selector != 'baseline':
        # Training redundant features
        new_features = int(args.redundancy*coords_per)
        parts = []
        for client in range(num_clients):
            gauss = np.random.normal(loc=np.average(train_raw[0]), scale=(train_raw[0].max()-train_raw[0].min())/2, size=(train_raw[0].shape[0], new_features))
            parts.append(np.concatenate((train_raw[0][:,coords_per*client:coords_per*(client+1)],
                                             gauss), axis=1))
        train_raw = (np.concatenate(parts, axis=1), train_raw[1])

        # Test redundant features
        parts = []
        for client in range(num_clients):
            gauss = np.random.normal(loc=np.average(test_raw[0]), scale=(test_raw[0].max()-test_raw[0].min())/2, size=(test_raw[0].shape[0], new_features))
            parts.append(np.concatenate((test_raw[0][:,coords_per*client:coords_per*(client+1)],
                                             gauss), axis=1))
        test_raw = (np.concatenate(parts, axis=1), test_raw[1])
        coords_per += new_features 

    args_dict = dict(args._get_kwargs())
    args_dict['task'] = 'ihm'
    args_dict['downstream_clients'] = num_clients # total number of vertical partitions present

    models = []
    embedding_size = args.embedding_size
    # Make models for each client
    for i in range(num_clients+1):
        # Build the model
        args.network = "mimic3models/keras_models/"
        if i < num_clients:
            args.network += "dense_bottom_sparse.py"
        else:
            args.network += "lstm_top_sparse.py"

        print("==> using model {}".format(args.network))
        model_module = imp.load_source(os.path.basename(args.network), args.network)
        if i < num_clients:
            model = model_module.Network(input_dim=coords_per, output_dim=embedding_size, lambd=args.lambda_client, **args_dict)
        else:
            model = model_module.Network(input_dim=coords_per, output_dim=embedding_size*num_clients, num_classes=num_classes, lambd=args.lambda_server, **args_dict)

        # Compile the model
        print("==> compiling the model")
        if args.optimizer == "SGD":
            optimizer_config = tf.keras.optimizers.SGD(learning_rate=args.lr)
        elif args.optimizer == "Adam":
            optimizer_config = tf.keras.optimizers.Adam(
                        learning_rate=args.lr, beta_1=args.beta_1)
        loss = 'binary_crossentropy'
        loss_weights = None
        if args.task == "regression":
            loss = 'mean_squared_error' 
        elif num_classes > 1:
            loss = 'categorical_crossentropy'

        model.compile(optimizer=optimizer_config,
                    loss=loss,
                    loss_weights=loss_weights)
        model.summary()
        models.append(model)

    # Load model weights
    n_trained_chunks = 0
    if args.load_state != "":
        model.load_weights(args.load_state)
        n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))


    # Prepare training

    print("==> training")

    activation = tf.keras.activations.sigmoid 

    # Training functions
    def get_grads(x, y, H, model, server_model, i, pretraining=False):
        """
        Calculate client embedding, and other client embeddings
        to get the gradient for client model
        """
        loss_value = 0
        Hnew = H.copy()
        with backprop.GradientTape() as tape:
            out = model(x, training=True)
            Hnew[i] = tf.squeeze(out)

            # Concatenate embeddings and get final loss
            logits = server_model(tf.concat(Hnew,axis=1), training=True)
            loss_value = server_model.compiled_loss(y, logits)
        grads = tape.gradient(loss_value, model.trainable_variables 
                                        + server_model.trainable_variables)
        return grads, loss_value

    def train_step(x, y, model, server_model, H, local, i, pretraining=False):
        """
        Training a client model 
        """
        grads, loss_value = get_grads(x, y, H, model, server_model, i, pretraining)
        model.optimizer.apply_gradients(zip(grads[:len(model.trainable_variables)],
                                            model.trainable_variables))
        return loss_value, grads[2][embedding_size*i:embedding_size*(i+1)]

    def getserver_grads(y, H, server_model, sparse=False):
        """
        Use client embeddings to get current server model gradient
        """
        loss_value = 0
        Hnew = H.copy()
        with backprop.GradientTape() as tape:
            # Concatenate embeddings and get final loss
            logits = server_model(tf.concat(Hnew,axis=1), training=True)
            loss_value = server_model.compiled_loss(y, logits)
        grads = tape.gradient(loss_value, server_model.trainable_variables)

        # Set NaN values to zero
        for g in range(len(grads)):
            grads[g] = tf.where(tf.math.is_nan(grads[g]), 0, grads[g]) 

        return grads, loss_value

    def trainserver_step(y, server_model, H, local, sparse=False, threshold=0):
        """
        Train the server model
        """
        global args
        grads, loss_value = getserver_grads(y, H, server_model, sparse)
        server_model.optimizer.apply_gradients(zip(grads,
                                        server_model.trainable_variables))
        if sparse:
            # Apply thresholding to the input layer weights for proximal gradient descent
            new_weights = server_model.layers[1].get_weights()
            removed = 0
            for feature in range(len(new_weights[0])):
                mag = tf.norm(new_weights[0][feature])
                if mag <= threshold:
                    new_weights[0][feature] = tf.zeros(new_weights[0][feature].shape)
                    removed += 1
                else:
                    new_weights[0][feature] -= threshold*new_weights[0][feature]/mag
            print(f"Server removed {removed} features")
            server_model.layers[1].set_weights(new_weights)
        return loss_value

    def get_gradsall(x, y, models, epoch=0, step=0):
        """
        Calculate client embedding, and other client embeddings
        to get the gradient for client model
        """
        loss_value = 0
        variables = []
        with backprop.GradientTape() as tape:
            out = []
            # Compute forward pass through VFL model
            for i in range(len(models)-1):
                x_local = x[:,coords_per*i:coords_per*(i+1)]
                x_local = tf.gather(x_local, x_local_inds[i], axis=1)
                out.append(models[i](x_local, training=True))
            logits = models[-1](tf.concat(out,axis=1), training=True)
            loss_value = models[-1].compiled_loss(y, logits)
            for model in models:
                variables += model.trainable_variables

        # Calculate gradients and remove NaN values
        grads = tape.gradient(loss_value, variables)
        for g in range(len(grads)):
            grads[g] = tf.where(tf.math.is_nan(grads[g]), 0, grads[g]) 
        return grads, loss_value

    def train_stepall(x, y, models, local, threshold_server, threshold_client, epoch=0, step=0):
        """
        Train full VFL model for one iteration 
        """
        grads, loss_value = get_gradsall(x, y, models, epoch, step)
        start = 0
        end = 0 
        for i, model in enumerate(models):
            end += len(model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads[start:end],
                                                model.trainable_variables))

            # Apply thresholding to the input layer weights for proximal gradient descent
            new_weights = model.layers[1].get_weights()
            removed = 0
            for feature in range(len(new_weights[0])):
                mag = tf.norm(new_weights[0][feature])
                if i != args.num_clients:
                    threshold = threshold_client
                else:
                    threshold = threshold_server
                if mag <= threshold:
                    new_weights[0][feature] = tf.zeros(new_weights[0][feature].shape)
                    removed += 1
                else:
                    new_weights[0][feature] -= threshold*new_weights[0][feature]/mag
            print(f"Client {i} removed {removed} features")
            model.layers[1].set_weights(new_weights)
            start = end
        return loss_value

    def get_grads_sparsify(x, out_true, model, i, keep_inds):#, error=False):
        """
        Calculate client embedding, and other client embeddings
        to get the gradient for client model
        """
        loss_value = 0
        # Compute forward pass through VFL model
        with backprop.GradientTape() as tape:
            out = model(x, training=True)
            loss_value = model.compiled_loss(tf.gather(out_true, keep_inds, axis=1), out)

        # Calculate gradients and remove NaN values
        grads = tape.gradient(loss_value, model.trainable_variables)
        for g in range(len(grads)):
            grads[g] = tf.where(tf.math.is_nan(grads[g]), 0, grads[g]) 
        return grads, loss_value

    def train_sparsify(x, y, model, i, keep_inds, threshold):
        """
        Traing a client model for 'local' local iterations
        """
        grads, loss_value = get_grads_sparsify(x, y, model, i, keep_inds)
        model.optimizer.apply_gradients(zip(grads,model.trainable_variables))

        # Apply thresholding to the input layer weights for proximal gradient descent
        new_weights = model.layers[1].get_weights()
        removed = 0
        for feature in range(len(new_weights[0])):
            mag = tf.norm(new_weights[0][feature])
            if mag <= threshold:
                new_weights[0][feature] = tf.zeros(new_weights[0][feature].shape)
                removed += 1
            else:
                new_weights[0][feature] -= threshold*new_weights[0][feature]/mag
        print(f"Client {i} removed {removed} features")
        model.layers[1].set_weights(new_weights)
        return loss_value

    # Get client embeddings
    def forward(x, y, model):
        out = model(x, training=False)
        return out 

    # Get predicted labels to calculating accuracy 
    def predict(x, y, models, x_local_inds):
        out = []
        for i in range(len(models)-1):
            x_local = x[:,coords_per*i:coords_per*(i+1)]
            out.append(forward(x_local, y, models[i]))
        logits = models[-1](tf.concat(out,axis=1), training=False)
        loss = models[-1].compiled_loss(y, logits)
        return logits , loss

    # Get predicted labels to calculating accuracy 
    def predict_sparse(x, y, models, remove_inds, x_local_inds, step, Hs_tmp=[]):
        out = []
        for i in range(len(models)-1):
            if len(x_local_inds[i]) == 0:
                out.append(Hs_tmp[step][i])
                continue
            x_local = x[:,coords_per*i:coords_per*(i+1)]
            x_local = tf.gather(x_local, x_local_inds[i], axis=1)
            outi = forward(x_local, y, models[i]).numpy()
            out.append(tf.squeeze(outi))
        logits = models[-1](tf.concat(out,axis=1), training=False)
        loss = models[-1].compiled_loss(y, logits)
        return logits , loss

    # Split data into batches 
    if args.dataset != 'modelnet':
        train_dataset = tf.data.Dataset.from_tensor_slices((
                                        train_raw[0], 
                                        train_raw[1]))
        train_dataset = train_dataset.batch(args.batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((
                                        test_raw[0], 
                                        test_raw[1]))
        test_dataset = test_dataset.batch(args.batch_size)

    def evaluate(models, losses, accs_train, accs_test, Hs=[], Hs_test=[]):
        """
        Print the current training/test loss and accuracy
        """
        if num_classes > 1:
            predictions = np.zeros((train_raw[0].shape[0], num_classes))
        else:
            predictions = np.zeros(train_raw[0].shape[0])
        left = 0
        total_loss = 0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            logits, loss_aggregate_model = predict_sparse(x_batch_train, y_batch_train, models, remove_inds, x_local_inds, step, Hs)
            total_loss += loss_aggregate_model
            if num_classes > 1:
                predictions[left: left + len(x_batch_train)] = logits
            else:
                predictions[left: left + len(x_batch_train)] = tf.reshape(tf.identity(logits),-1)
            left = left + len(x_batch_train)
        losses.append(total_loss/len(train_dataset))
        print(f"************Training Loss = {losses[-1]}***************")
        pickle.dump(losses, open(f'losses{suffix}.pkl', 'wb'))

        # Calculate Training Accuracy 
        if args.task == "classification":
            if math.isnan(losses[-1]):
                accs_train.append([0])
            else:
                if num_classes > 1:
                    ret = metrics.print_metrics_multilabel(train_raw[1], predictions, verbose=0)
                else:
                    ret = metrics.print_metrics_binary(train_raw[1], predictions, verbose=0)
                ret_preds = predictions
                accs_train.append(list(ret.items()))
                print(f"************Train Accuracy = {ret['acc']}************")
                if num_classes == 1:
                    print(f"************Train F1-Score = {ret['f1_score']}************")
            pickle.dump(accs_train, open(f'accs_train{suffix}.pkl', 'wb'))

        # Calculate Test Accuracy 
        if num_classes > 1:
            predictions = np.zeros((test_raw[0].shape[0], num_classes))
        else:
            predictions = np.zeros(test_raw[0].shape[0])
        left = 0
        total_loss = 0
        for step, (x_batch_test, y_batch_test) in enumerate(test_dataset):
            logits, loss_aggregate_model = predict_sparse(x_batch_test, y_batch_test, models, remove_inds, x_local_inds, step, Hs_test)
            total_loss += loss_aggregate_model
            if num_classes > 1:
                predictions[left: left + len(x_batch_test)] = logits
            else:
                predictions[left: left + len(x_batch_test)] = tf.reshape(tf.identity(logits),-1)
            left = left + len(x_batch_test)
        print(f"************Test Loss = {total_loss}***************")

        if args.task == "classification":
            if math.isnan(total_loss):
                accs_test.append([0])
            else:
                if num_classes > 1:
                    ret = metrics.print_metrics_multilabel(test_raw[1], predictions, verbose=0)
                else:
                    ret = metrics.print_metrics_binary(test_raw[1], predictions, verbose=0)
                accs_test.append(list(ret.items()))
                print(f"************Test Accuracy = {ret['acc']}************")
                if num_classes == 1:
                    print(f"************Test F1-Score = {ret['f1_score']}************")
            pickle.dump(accs_test, open(f'accs_test{suffix}.pkl', 'wb'))
        
        # Print the percentage of spurious features removed
        if args.redundancy > 0:
            start = input_size[args.dataset]
            end = int(input_size[args.dataset]*(1+args.redundancy))
            correct = 0
            total = 0
            for client in range(args.num_clients):
                new_weights = models[client].layers[1].weights
                new_weights = [w.numpy() for w in new_weights]
                importance = K.sqrt(K.sum(K.square(new_weights[0]), axis=1))
                zero = np.where(importance <= threshold)[0]
                red = np.arange(start,end)
                correct += len(list(set(red).intersection(zero)))
                total += len(red)
            print(f"************Spurious Features Removed = {(correct/total)*100}************")

    workers = num_clients
    losses = []
    accs_train = []
    accs_test = []
    x_local_inds = []
    embedding_comp = []
    remove_comp = []
    # Initialize vectors for keeping track of features and embedding components indices
    for i in range(num_clients):
        x_local_inds.append(np.arange(0,coords_per))
        embedding_comp.append(np.arange(0,embedding_size))
        remove_comp.append([])
    remove_inds = []

    # Get initial loss and accuracy
    evaluate(models, losses, accs_train, accs_test)

    grads_Hs = np.empty((num_clients), dtype=object)
    grads_Hs.fill([])
    server_model = models[-1]
    # Main training loop
    for epoch in range(args.epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        Hs = np.empty((math.ceil(train_raw[0].shape[0] / args.batch_size), num_clients), dtype=object)
        Hs.fill([])
        num_batches = 0

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_dataset)):
            num_batches += 1

            # Exchange client embeddings
            for i in range(num_clients):
                x_local = x_batch_train[:,coords_per*i:coords_per*(i+1)]
                H_out = forward(x_local, y_batch_train, models[i])
                Hs[step,i] = copy.deepcopy(H_out)

            # Train for each client 
            client_losses = [0]*num_clients
            for i in range(num_clients):
                x_local = x_batch_train[:,coords_per*i:coords_per*(i+1)]
                H = copy.deepcopy(Hs[step]).tolist()

                le = local_epochs
                client_losses[i], grads_Hs[i] = train_step(x_local, y_batch_train, models[i], 
                                              server_model, H, le, i)

            H = copy.deepcopy(Hs[step]).tolist()
            # Train server
            loss_final = trainserver_step(y_batch_train, models[-1], H, local_epochs)

        print("==> predicting")
        evaluate(models, losses, accs_train, accs_test)
        end_time = time.time()
        print(end_time - start_time)

    # Get most recent trained embeddings
    Hs = np.empty((math.ceil(train_raw[0].shape[0] / args.batch_size), num_clients), dtype=object)
    Hs_cat = np.empty((math.ceil(train_raw[0].shape[0] / args.batch_size)), dtype=object)
    Hs.fill([])
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # Exchange client embeddings
        for i in range(num_clients):
            x_local = x_batch_train[:,coords_per*i:coords_per*(i+1)]
            x_local = tf.gather(x_local, x_local_inds[i], axis=1)
            H_out = forward(x_local, y_batch_train, models[i])
            Hs[step,i] = copy.deepcopy(H_out)
        Hs_cat[step] = tf.concat(Hs[step].tolist(), axis=1)
        for i in range(num_clients):
            Hs[step,i] = Hs[step,i].numpy() 
    Hs_comb = tf.concat(Hs_cat.tolist(), axis=0)

    # Get most recent test embeddings
    Hs_test = np.empty((len(test_dataset), num_clients), dtype=object)
    Hs_test.fill([])
    # Iterate over the batches of the dataset.
    for step, (x_batch_test, y_batch_test) in enumerate(test_dataset):
        # Exchange client embeddings
        for i in range(num_clients):
            x_local = x_batch_test[:,coords_per*i:coords_per*(i+1)]
            x_local = tf.gather(x_local, x_local_inds[i], axis=1)
            H_out = forward(x_local, y_batch_test, models[i])
            Hs_test[step,i] = copy.deepcopy(H_out)
        for i in range(num_clients):
            Hs_test[step,i] = Hs_test[step,i].numpy() 

    importances_all = []
    if args.selector == 'emb_sparse' or args.selector == 'centralized':
        # Start embedding component selection for LESS-VFL (emb_sparse)
        # Or start group lasso feature selection (centralized)

        # Reinitialize models with new optimizer 
        loss = 'binary_crossentropy'
        loss_weights = None
        if args.task == "regression":
            loss = 'mean_squared_error' 
        elif num_classes > 1:
            loss = 'categorical_crossentropy'

        optimizer_config = tf.keras.optimizers.SGD(learning_rate=0.01)

        for i in range(num_clients+1):
            models[i].compile(optimizer=optimizer_config,
                        loss=loss,
                        loss_weights=loss_weights)

        # Sparsify server model training loop 
        # Or group lasso training loop
        for epoch in range(150):
            start_time = time.time()
            print("\nStart of epoch %d for sparsification" % (epoch,))
            num_batches = 0

            # Iterate over the batches of the dataset.
            overall_loss = 0
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                if args.selector == 'centralized':
                    # Group lasso with proximal SGD
                    loss = train_stepall(x_batch_train, y_batch_train, models, 1, args.lambda_server*args.lr, args.lambda_client*args.lr, epoch, step)
                else:
                    # Sparsify server model without communication
                    loss = trainserver_step(y_batch_train, server_model, Hs[step].tolist(), 1, sparse=True, threshold=args.lambda_server*args.lr)
                overall_loss += loss

            print(f"************Sparsification Loss = {overall_loss}***************")
            print("==> predicting")
            evaluate(models, losses, accs_train, accs_test)
            end_time = time.time()
            print(end_time - start_time)

            # Save current input model weights
            if args.selector == 'centralized':
                importances = []
                for client in range(args.num_clients):
                    new_weights = models[client].layers[1].weights
                    new_weights = [w.numpy() for w in new_weights]
                    importance = K.sqrt(K.sum(K.square(new_weights[0]), axis=1))
                    remove_inds = np.where(np.array(importance) < threshold)[0]
                    importances.append(remove_inds)
                importances_all.append(importances)

        # Get current input model weights
        # Determine indices of embedding components to keep/remove
        if args.selector == 'emb_sparse':
            new_weights = models[-1].layers[1].weights
            new_weights = [w.numpy() for w in new_weights]
            importance = K.sqrt(K.sum(K.square(new_weights[0]), axis=1))
            emb_keep_inds = np.where(np.array(importance) >= threshold)[0]
            remove_inds = np.where(np.array(importance) < threshold)[0]
        else:
            importance = []
            remove_inds = []
            emb_keep_inds = np.arange(embedding_size*num_clients)

    else:
        importance = []
        remove_inds = [] 
        emb_keep_inds = np.arange(embedding_size*num_clients)

    print(remove_inds)
    print(f"Removing {len(remove_inds)} embedding components")

    # Offset embedding component indices list to match client embedding indices
    offset = 0
    for client in range(num_clients):
        inds = emb_keep_inds-offset
        inds = [ind for ind in inds if ind in np.arange(0,len(embedding_comp[client]))] 
        offset += len(embedding_comp[client])
        embedding_comp[client] = inds

    # Create server model with reduced number of embedding components
    args.network = "mimic3models/keras_models/"
    args.network += "lstm_top_sparse.py"

    print("==> using model {}".format(args.network))
    model_module = imp.load_source(os.path.basename(args.network), args.network)
    model = model_module.Network(input_dim=coords_per, output_dim=len(emb_keep_inds), num_classes=num_classes, **args_dict)

    # Compile the model
    print("==> compiling the model")
    if args.optimizer == "SGD":
        optimizer_config = tf.keras.optimizers.SGD(learning_rate=args.lr)
    elif args.optimizer == "Adam":
        optimizer_config = tf.keras.optimizers.Adam(
                learning_rate=args.lr, beta_1=args.beta_1)
    loss = 'binary_crossentropy'
    loss_weights = None
    if args.task == "regression":
        loss = 'mean_squared_error' 
    elif num_classes > 1:
        loss = 'categorical_crossentropy'

    model.compile(optimizer=optimizer_config,
                loss=loss,
                loss_weights=loss_weights)
    new_weights = models[-1].layers[-1].weights
    new_weights = [w.numpy() for w in new_weights]
    new_weights[0] = new_weights[0][emb_keep_inds,:] 
    model.layers[-1].set_weights(new_weights)
    models[-1] = model

    if args.selector == 'emb_sparse' or args.selector == 'sparse':
        # Clients run optimization to sparsify local network
        for client in range(num_clients):
            # Recompile networks with SparsifyLoss:
            # difference between pre-trained embeddings and current 
            args.network = "mimic3models/keras_models/"
            args.network += "dense_bottom_sparse.py"

            model_module = imp.load_source(os.path.basename(args.network), args.network)
            model = model_module.Network(input_dim=coords_per, 
                                         output_dim=len(embedding_comp[client]), 
                                         **args_dict)

            # Compile the model
            if args.dataset == 'mimic3':
                optimizer_config = tf.keras.optimizers.SGD(learning_rate=0.0001)
            else:
                optimizer_config = tf.keras.optimizers.SGD(learning_rate=0.001)
            loss_weights = None
            model.compile(optimizer=optimizer_config,
                        loss=SparsifyLoss(),
                        loss_weights=loss_weights)

            # Copy model weights over
            for l in range(1, len(models[client].layers)-1):
                model.layers[l].set_weights(models[client].layers[l].get_weights())
            if len(models[client].layers) > 2:
                new_weights = models[client].layers[-1].weights
            else:
                new_weights = model.layers[-1].weights
            new_weights = [w.numpy() for w in new_weights]
            new_weights[0] = new_weights[0][:,embedding_comp[client]] 
            new_weights[1] = new_weights[1][embedding_comp[client]]
            model.layers[-1].set_weights(new_weights)
            models[client] = model

        # Client sparsify training loop 
        for epoch in range(150): 
            start_time = time.time()
            print("\nStart of epoch %d for sparsification" % (epoch,))
            num_batches = 0

            # Iterate over the batches of the dataset.
            overall_loss = 0
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                num_batches += 1

                # Sparsify client models without communication
                for i in range(num_clients):
                    x_local = x_batch_train[:,coords_per*i:coords_per*(i+1)]
                    x_local = tf.gather(x_local, x_local_inds[i], axis=1)
                    client_losses[i] = train_sparsify(x_local, tf.convert_to_tensor(Hs[step,i]), 
                                                        models[i], i, embedding_comp[i], threshold=args.lambda_client*args.lr)
                    overall_loss += client_losses[i]

            print(f"************Sparsification Loss = {overall_loss}***************")
            print("==> predicting")
            evaluate(models, losses, accs_train, accs_test)
            end_time = time.time()
            print(end_time - start_time)

    # Determine which features to remove
    features_removed = []
    importances = []
    for client in range(args.num_clients):
        if args.selector == 'baseline' or args.selector == 'baseline_red':
            break

        new_weights = models[client].layers[1].weights
        new_weights = [w.numpy() for w in new_weights]
        if args.selector == 'emb_sparse' or args.selector == 'sparse' or args.selector == 'centralized':
            keep_inds = []
            del_inds = []
            weight_med = np.mean(new_weights[0])
            importance = []
            importance = K.sqrt(K.sum(K.square(new_weights[0]), axis=1))
            keep_inds = np.where(np.array(importance) >= threshold_client)[0]
            del_inds = np.where(np.array(importance) < threshold_client)[0]
        elif args.selector == 'random':
            indices = np.random.permutation(coords_per)
            keep = int(coords_per*args.feature_per)
            keep_inds = indices[-keep:]
            del_inds = indices[:-keep]
        else:
            keep_inds = (keep_inds_all[(keep_inds_all >= client*coords_per) 
                                        & (keep_inds_all < (client+1)*coords_per)]
                            - client*coords_per)
            del_inds = (del_inds_all[(del_inds_all >= client*coords_per) 
                                        & (del_inds_all < (client+1)*coords_per)]
                            - client*coords_per)

        to_del = len(del_inds)
        print(f"Client {client}: Deleting {to_del} feature(s)")
        print(del_inds+coords_per*client)
        features_removed.append(del_inds+coords_per*client)
        importances.append(importance)

        x_local_inds[client] = keep_inds
        new_weights[0] = np.delete(new_weights[0], del_inds, axis=0)

        # Create new network for training with remaining features 
        args.network = "mimic3models/keras_models/"
        args.network += "dense_bottom_sparse.py"

        model_module = imp.load_source(os.path.basename(args.network), args.network)
        model = model_module.Network(input_dim=len(keep_inds), 
                                     output_dim=len(embedding_comp[client]), 
                                     **args_dict)

        # Compile the model
        if args.optimizer == "SGD":
            optimizer_config = tf.keras.optimizers.SGD(learning_rate=args.lr)
        elif args.optimizer == "Adam":
            optimizer_config = tf.keras.optimizers.Adam(
                    learning_rate=args.lr, beta_1=args.beta_1)
        loss = 'binary_crossentropy'
        loss_weights = None
        if args.task == "regression":
            loss = 'mean_squared_error' 
        elif num_classes > 1:
            loss = 'categorical_crossentropy'

        model.compile(optimizer=optimizer_config,
                    loss=loss,
                    loss_weights=loss_weights)

        for l in range(2, len(models[client].layers)):
            model.layers[l].set_weights(models[client].layers[l].get_weights())
        model.layers[1].set_weights(new_weights)
            
        models[client] = model

    # Save which features are deleted
    pickle.dump(features_removed, open(f'features{suffix}.pkl', 'wb'))
    pickle.dump(importances, open(f'importances{suffix}.pkl', 'wb'))

    if args.selector == 'centralized':
        importances_all.append(importances)
        pickle.dump(importances_all, open(f'importances_all{suffix}.pkl', 'wb'))
    if args.selector == 'emb_sparse':
        pickle.dump(embedding_comp, open(f'embeddings_removed{suffix}.pkl', 'wb'))

    # Get most recent trained embeddings
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        for i in range(num_clients):
            if len(x_local_inds[i]) == 0:
                Hs[step,i] = tf.squeeze(Hs[step,i][:,embedding_comp[i]])

    # Get most recent test embeddings
    # Iterate over the batches of the dataset.
    for step, (x_batch_test, y_batch_test) in enumerate(test_dataset):
        for i in range(num_clients):
            if len(x_local_inds[i]) == 0:
                Hs_test[step,i] = tf.squeeze(Hs_test[step,i][:,embedding_comp[i]])

    grads_Hs = np.empty((num_clients), dtype=object)
    grads_Hs.fill([])
    # Post feature selection training 
    for epoch in range(100): 
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        num_batches = 0

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_dataset)):
            num_batches += 1

            # Exchange client embeddings
            for i in range(num_clients):
                if len(x_local_inds[i]) == 0:
                    continue
                x_local = x_batch_train[:,coords_per*i:coords_per*(i+1)]
                x_local = tf.gather(x_local, x_local_inds[i], axis=1)
                H_out = forward(x_local, y_batch_train, models[i])
                Hs[step,i] = copy.deepcopy(tf.squeeze(H_out))

            server_model = models[-1]

            # Train for each client 
            client_losses = [0]*num_clients
            for i in range(num_clients):
                if len(x_local_inds[i]) == 0:
                    continue
                x_local = x_batch_train[:,coords_per*i:coords_per*(i+1)]
                x_local = tf.gather(x_local, x_local_inds[i], axis=1)
                H = copy.deepcopy(Hs[step]).tolist()

                le = local_epochs
                client_losses[i], grads_Hs[i] = train_step(x_local, y_batch_train, models[i], 
                                              server_model, H, le, i)

            H = copy.deepcopy(Hs[step]).tolist()
            # Train server
            loss_final = trainserver_step(y_batch_train, models[-1], H, local_epochs)

        print("==> predicting")
        evaluate(models, losses, accs_train, accs_test, Hs, Hs_test)
        end_time = time.time()
        print(end_time - start_time)

