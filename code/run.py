import argparse
import glob
import json
from os import mkdir, path

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

def get_scorer(is_categorical):

    return 
    if is_categorical:
         mcc()
    else:
        scorer = "neg_root_mean_squared_error"

parser = argparse.ArgumentParser()

#parser.add_argument(
#    '-m',
#    '--model',
#    dest='model',
#    action='store',
#    required=True,
#    help='Type of model considered'
#)
#parser.add_argument(
#    '-c',
#    '--classification',
#    dest='model_type',
#    action='store',
#    required=True,
#    help='True if classification false if regression'
#)
#parser.add_argument(
#    '--dataset',
#    '-d',
#    dest='dataset',
#    action='store',
#    required=True,
#    help='Name or path of the dataset to be considered'
#)
#parser.add_argument(
#    '--outputFile',
#    '-o',
#    dest='results_file',
#    action='store',
#    required=True,
#    help='Path to the file where the results are stored (.csv file)'
#)

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
    #params["classifier_random_state"] = something

    if dataset in jobs:

        params["classifier"] = [get_model(params["model"], params["is_categorical"])]

        del params["dataset"]
        del params["is_categorical"]
        del params["model"]

        jobs[dataset]["hyper_set"].append(params) 

    else:

        params["classifier"] = [get_model(params["model"], params["is_categorical"])]
        scorer = mcc() if params["is_categorical"] else "neg_root_mean_squared_error"
        output = params["output"]

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
        mkdir(results_dir)

    '''
        Run search
    '''
    search(jobs[dataset]["hyper_set"], x_train, y_train, scorer, jobs[dataset]["output"], dataset)

