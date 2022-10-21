import argparse
import glob
import json
import models
import datasets
from keras.utils import to_categorical
import numpy as np
from search import search
from metrics import mcc

parser = argparse.ArgumentParser()

parser.add_argument(
    '--hyper',
    dest='hyper_path',
    action='store',
    required=True,
    help='Folder with multiple hyperparameter sets'
)

parser.add_argument(
    '-m',
    '--model',
    dest='model',
    action='store',
    required=True,
    help='Type of model considered'
)

parser.add_argument(
    '-c',
    '--classification',
    dest='model_type',
    action='store',
    required=True,
    help='True if classification false if regression'
)

parser.add_argument(
    '--outputFile',
    '-o',
    dest='results_file',
    action='store',
    required=True,
    help='Path to the file where the results are stored (.csv file)'
)

parser.add_argument(
    '--dataset',
    '-d',
    dest='dataset',
    action='store',
    required=True,
    help='Name of the dataset to be considered'
)

parser.add_argument(
    '--datasetFile',
    dest='datasetFile',
    action='store',
    required=False,
    help='File of the dataset to be loaded'
)

args = parser.parse_args()

print('Creating the model')
if args.model == "dense":
    model = models.create_dense_model(bool(args.model_type))
elif args.model == "cnn":
    print("entrei")
    model = models.create_cnn_model(bool(args.model_type))
else:
    model = models.create_gru_model(bool(args.model_type))

print('Choosing metrics')
if bool(args.model_type):
    scorer = mcc()
else:
    scorer = ""         # needs to be defined
 

print('Loading Hyperparameter sets.')
hyper_files = glob.glob(args.hyper_path+'*.json')
hyper_set = []

for f in hyper_files:
    params = json.load(open(f))
    params["classifier"] = [model]
    hyper_set.append(params)


print('Loading the dataset considered')
if args.dataset == "cifar10":
    train_data, test_data= datasets.load_cifar10()

    x_train, y_train = train_data

    y_train = to_categorical(y_train)

    if args.model == "dense":
        x_train = np.reshape(x_train[:,:,:,0], (x_train.shape[0], -1))
    elif args.model == "gru":
        x_train = x_train[:,:,:,0]
        
elif args.dataset == "fashion_mnist":
    train_data, test_data= datasets.load_fashion_mnist()

    x_train, y_train = train_data

    y_train = to_categorical(y_train)

else:
    train_data, test_data= datasets.load_dataset(args.datasetFile)

    x_train, y_train = train_data

    y_train = to_categorical(y_train)



search(hyper_set, x_train, y_train, scorer, args.results_path)

