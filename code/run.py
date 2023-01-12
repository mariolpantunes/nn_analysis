import argparse
import glob
import json
from os import mkdir, path

import models
import numpy as np
from keras.utils import to_categorical
from metrics import mcc
from search import search

import datasets

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


parser.add_argument(
    '--hyper',
    dest='hyper_path',
    action='store',
    required=True,
    help='Folder with multiple hyperparameter sets' 
)

parser.add_argument(
    '--outputFile',
    '-o',
    dest='results_file',
    action='store',
    required=True,
    help='Path to the file where the results are stored (.csv file)'
)

args = parser.parse_args()

'''
    Load hyperparameter sets for evaluation
'''
if not path.exists(args.hyper_path):
    raise ValueError("Hyperparameter folder does not exist.")

hyper_files = glob.glob(args.hyper_path+'*.json')
hyper_set = []

for f in hyper_files:
    params = json.load(open(f))
    params["classifier"] = [model]
    hyper_set.append(params)
    
if not hyper_set:
    raise ValueError("Hyperparameter folder is empty. \n Ensure that every hyperparameter set is a json file.")


'''
    Check if is classification or regression
'''
classification = args.model_type == "True"
if classification:
    scorer = mcc()
else:
    scorer = "neg_root_mean_squared_error"


'''
    Create default model
'''
if args.model == "dense":
    model = models.create_dense_model(classification)
elif args.model == "cnn":
    model = models.create_cnn_model(classification)
elif args.model == "gru":
    model = models.create_gru_model(classification)
else:
    raise ValueError("Model should be either 'dense', 'cnn', or 'gru'")


'''
    Load the dataset
'''
if args.dataset == "cifar10":
    train_data, test_data= datasets.load_cifar10()

    x_train, y_train = train_data

    if classification:
        y_train = to_categorical(y_train)

    if args.model == "dense":                   # needs to be corrected for a standard preprocessing
        x_train = np.reshape(x_train[:,:,:,0], (x_train.shape[0], -1))
    elif args.model == "gru":
        x_train = x_train[:,:,:,0]
        
elif args.dataset == "fashion_mnist":
    train_data, test_data= datasets.load_fashion_mnist()

    x_train, y_train = train_data

    if classification:
        y_train = to_categorical(y_train)

else:
    if not path.exists(args.dataset):
        raise ValueError("Dataset file does not exist. \n the names of the default datasets are 'cifar10', 'fashion_mnist'")
    train_data, test_data= datasets.load_dataset(args.dataset)

    x_train, y_train = train_data

    if classification:
        y_train = to_categorical(y_train)
    
'''
    Results folder creation
'''
results_dir = "/".join(args.results_file.split("/"))
if not path.exists(results_dir):
    mkdir(results_dir)


'''
    Run search
'''

search(hyper_set, x_train, y_train, scorer, args.results_file)

