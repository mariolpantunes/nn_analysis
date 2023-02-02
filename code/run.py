import argparse
import glob
import json
from os import makedirs, path

import models
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from metrics import mcc
from search import search

import datasets

tf.keras.utils.set_random_seed(1)

def get_model(model_name, is_categorical):
    '''
    Create default model
    '''
    if model_name == "dense":
        return models.create_dense_model(is_categorical)
    elif model_name == "cnn":
        return models.create_cnn_model(is_categorical)
    elif model_name == "rnn":
        return models.create_rnn_model(is_categorical)
    
    raise ValueError("Model should be either 'dense', 'cnn', or 'gru'")

def check_hyperparameters(hyperparameters, f):
    if hyperparameters['model'] == "dense":
        for hidden_layers in hyperparameters['classifier__hidden_layer_dims']:
            for activation_functions in hyperparameters['classifier__activation_functions']:
                if len(hidden_layers) != len(activation_functions):
                    raise ValueError("The number of hidden layers should be equal to the number of activation functions"+
                        ", please check file: " + f)

            for dropouts in hyperparameters["classifier__dropouts"]:
                if len(dropouts) != len(hidden_layers):
                    raise ValueError("The number of hidden layers should be equal to the number of dropouts. \n"+
                        "if you don't want dropouts set them to zero, please check file: " + f)

            for hidden_layer in hidden_layers:
                if hidden_layer <= 0:
                    raise ValueError("Invalid hidden layer dimension it should be >= 1"+
                    ", please check file: " + f)  

        for dropouts in hyperparameters["classifier__dropouts"]: 
            for dropout in dropouts:
                if dropout < 0 or dropout > 1:
                    raise ValueError("Invalid dropout value, the dropout should be in the interval [0, 1["+
                    ", please check file: " + f)
                  

    elif hyperparameters['model']  == "cnn":
        for n_kernels in hyperparameters["classifier__n_kernels"]:
            for kernel_sizes in hyperparameters["classifier__kernel_sizes"]:
                if len(n_kernels) != len(kernel_sizes):
                    raise ValueError("The 'n_kernels' list should have the same size as the 'kernel_sizes'"+
                        ", please check file: " + f)
                
            for pool_sizes in hyperparameters["classifier__pool_sizes"]:
                if len(n_kernels) != len(pool_sizes):
                    raise ValueError("The 'n_kernels' list should have the same size as the 'pool_sizes'"+
                        ", please check file: " + f)

            for activation_functions in hyperparameters["classifier__activation_functions"]:
                if len(n_kernels) != len(activation_functions):
                    raise ValueError("The 'n_kernels' list should have the same size as the 'activation_functions'"+
                        ", please check file: " + f)
            
            for dropouts in hyperparameters["classifier__dropouts"]:
                if len(dropouts) != len(n_kernels):
                    raise ValueError("The 'n_kernels' list should have the same size as the 'dropouts'"+
                    ", please check file: " + f)
            
        for kernel_sizes in hyperparameters["classifier__kernel_sizes"]:
            for kernel_size in kernel_sizes:
                if len(kernel_size) != hyperparameters["classifier__dimension"][0]:
                    raise ValueError("The kernels dimensions in the 'kernel_sizes' list should have the same dimensions as the 'dimension'"+
                        ", please check file: " + f)

        for dropouts in hyperparameters["classifier__dropouts"]: 
            for dropout in dropouts:
                if dropout < 0 or dropout > 1:
                    raise ValueError("Invalid dropout value, the dropout should be in the interval [0, 1["+
                    ", please check file:" + f)

        for dense_sizes in hyperparameters["classifier__dense_sizes"]:
            for dense_activations in hyperparameters["classifier__dense_activations"]:
                if len(dense_sizes) != len(dense_activations):
                    raise ValueError("The 'dense_sizes' list should have the same size as the 'dense_activations'"+
                    ", please check file: " + f)

            for dense_dropouts in hyperparameters["classifier__dense_dropouts"]:
                if len(dense_sizes) != len(dense_dropouts):
                    raise ValueError("The 'dense_sizes' list should have the same size as the 'dense_dropouts'"+
                        ", please check file: " + f)
        
        for dropouts in hyperparameters["classifier__dense_dropouts"]: 
            for dropout in dropouts:
                if dropout < 0 or dropout > 1:
                    raise ValueError("Invalid dropout value, the dropout should be in the interval [0, 1["+
                    ", please check file: " + f)
        
    else:
        for hidden_layers in hyperparameters['classifier__hidden_layer_dims']:
            for activation_functions in hyperparameters['classifier__activation_functions']:
                if len(hidden_layers) != len(activation_functions):
                    raise ValueError("The number of hidden layers should be equal to the number of activation functions\
                        , please check file: " + f)

            for dropouts in hyperparameters["classifier__dropouts"]:
                if len(dropouts) != len(hidden_layers):
                    raise ValueError("The number of hidden layers should be equal to the number of dropouts. \n\
                        if you don't want dropouts set them to zero, please check file:" + f)

            for hidden_layer in hidden_layers:
                if hidden_layer <= 0:
                    raise ValueError(" Invalid hidden layer dimension it should be >= 1" +
                    ", please check file: " + f)  
        

        for dropouts in hyperparameters["classifier__dropouts"]: 
            for dropout in dropouts:
                if dropout < 0 or dropout > 1:
                    raise ValueError("Invalid dropout value, the dropout should be in the interval [0, 1["+
                    ", please check file: " + f)

        for dense_sizes in hyperparameters["classifier__dense_sizes"]:
            for dense_activations in hyperparameters["classifier__dense_activations"]:
                if len(dense_sizes) != len(dense_activations):
                    raise ValueError("The 'dense_sizes' list should have the same size as the 'dense_activations'"+
                    ", please check file: " + f)

            for dense_dropouts in hyperparameters["classifier__dense_dropouts"]:
                if len(dense_sizes) != len(dense_dropouts):
                    raise ValueError("The 'dense_sizes' list should have the same size as the 'dense_dropouts'"+
                        ", please check file: " + f)
        
        for dropouts in hyperparameters["classifier__dense_dropouts"]: 
            for dropout in dropouts:
                if dropout < 0 or dropout > 1:
                    raise ValueError("Invalid dropout value, the dropout should be in the interval [0, 1["+
                    ", please check file: " + f)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--hyper',
    dest='hyper_path',
    action='store',
    required=True,
    help='Folder with multiple hyperparameter sets' 
)

args = parser.parse_args()

'''
    Load hyperparameter sets for evaluation
'''
if not path.exists(args.hyper_path):
    raise ValueError("Hyperparameter folder does not exist.")

hyper_files = glob.glob(args.hyper_path+'*.json')

if not hyper_files:
    raise ValueError("Hyperparameter folder is empty. \n Ensure that every hyperparameter set is a json file.")

jobs = {} #gets a set of models to run for each dataset

for f in hyper_files:
    params = json.load(open(f))
    dataset = params["dataset"]

    if dataset in jobs:

        params["classifier"] = [get_model(params["model"], params["is_categorical"])]

        check_hyperparameters(params, f)

        del params["dataset"]
        del params["is_categorical"]
        del params["model"]
        del params["output"]
        
        jobs[dataset]["hyper_set"].append(params) 

    else:

        params["classifier"] = [get_model(params["model"], params["is_categorical"])]
        scorer = mcc() if params["is_categorical"] else "neg_root_mean_squared_error"
        output = params["output"]

        check_hyperparameters(params, f)

        del params["dataset"]
        del params["is_categorical"]
        del params["model"]
        del params["output"]

        jobs[dataset] = {"hyper_set" : [params], "scorer" : scorer, "output" : output}


'''
    Load the dataset and run experiment
'''
for dataset in jobs:

    if dataset == "CIFAR10":
        train_data, test_data = datasets.load_cifar10()
    else:
        if not path.exists(dataset):
            raise ValueError("Dataset file does not exist.")
        train_data, test_data= datasets.load_dataset(dataset)

    x_train, y_train = train_data
    #x_train = x_train[:,:,:,0]


    if str(jobs[dataset]["scorer"]) == "make_scorer(mcc)" :
        y_train = to_categorical(y_train)
    
    '''
        Results folder creation
    '''
    results_dir = "/".join(jobs[dataset]["output"].split("/")[:-1])
    if not path.exists(results_dir):
        makedirs(results_dir)

    '''
        Run search
    '''
    search(jobs[dataset]["hyper_set"], x_train, y_train, scorer, jobs[dataset]["output"], dataset)

