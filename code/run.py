import argparse
import gc
import glob
import json
from os import makedirs, path

import dataset_loader
import models
import tensorflow as tf
from search import search
from sklearn.utils import shuffle

tf.keras.utils.set_random_seed(1)

'''
Missing correcting for verifications
'''
def check_hyperparameters(hyperparameters, f):#elu, 
    activation_functions = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", 
            "leaky_relu", "relu6", "silu", "gelu", "hard_sigmoid", "linear", "mish", "log_softmax"]
    
    optimizers = ["Adam", "SGD", "RMSprop", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"]

    losses = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", 
              "mean_squared_logarithmic_error", "cosine_similarity", "huber_loss", "log_cosh",
              "sparse_categorical_crossentropy", "binary_crossentropy", "categorical_crossentropy"]

    for activation in hyperparameters["dense_activation_fn"]:
        if activation not in activation_functions:
                    raise ValueError(f"Invalid activation function - {activation}, it should be one of the supported values {activation_functions}"+
                    ", please check file: " + f)
        
    if hyperparameters["batch_size"][0] < 1 :
            raise ValueError("Invalid minimum batch size, it should be a positive integer"+
            ", please check file: " + f)
        
    if hyperparameters["batch_size"][0] >  hyperparameters["batch_size"][1]:
                raise ValueError("Invalid maximum batch size, it should be a superior than the minimum"+
                ", please check file: " + f)

    if hyperparameters["batch_size"][1] - hyperparameters["batch_size"][0] < hyperparameters["batch_size"][2] :
                raise ValueError("Invalid step of batch size, it should be at least equal to maximum - minimum"+
                ", please check file: " + f)
    
    if hyperparameters["lr"][0] <= 0 :
            raise ValueError("Invalid minimum learning rate, it should be a positive float"+
            ", please check file: " + f)
    
    if hyperparameters["lr"][0] >  hyperparameters["lr"][1]:
                raise ValueError("Invalid maximum learning rate, it should be a superior than the minimum"+
                ", please check file: " + f)

    if hyperparameters["lr"][1] - hyperparameters["lr"][0] < hyperparameters["lr"][2] :
                raise ValueError("Invalid step of learning rate, it should be at least equal to maximum - minimum"+
                ", please check file: " + f)
    
    for optimizer in hyperparameters["optimizer"]:
        if optimizer not in optimizers:
                    raise ValueError(f"Invalid optimizer - {optimizer}, it should be one of the supported values {optimizers}"+
                    ", please check file: " + f)
    
    for loss in hyperparameters["loss"]:
        if loss not in losses:
                    raise ValueError(f"Invalid loss - {loss}, it should be one of the supported values {losses}"+
                    ", please check file: " + f)
        
    if hyperparameters["n_dense_layers"][0] < 1 :
            raise ValueError("Invalid minimum Nº of hidden layers, it should be a positive integer"+
            ", please check file: " + f)
    
    if hyperparameters["n_dense_layers"][0] >  hyperparameters["n_dense_layers"][1]:
                raise ValueError("Invalid maximum Nº of hidden layers, it should be a superior than the minimum"+
                ", please check file: " + f)

    if hyperparameters["lr"][1] - hyperparameters["lr"][0] < hyperparameters["lr"][2] :
            raise ValueError("Invalid step for the Nº of hidden layers, it should be at least equal to maximum - minimum"+
            ", please check file: " + f)
            
    if hyperparameters["n_dense_nodes"][0] < 1 :
            raise ValueError("Invalid minimum Nº of nodes, it should be a positive integer"+
            ", please check file: " + f)
    
    if hyperparameters["n_dense_nodes"][0] >  hyperparameters["n_dense_nodes"][1]:
                raise ValueError("Invalid maximum Nº of nodes, it should be a superior than the minimum"+
                ", please check file: " + f)

    if hyperparameters["n_dense_nodes"][1] - hyperparameters["n_dense_nodes"][0] < hyperparameters["n_dense_nodes"][2] :
                raise ValueError("Invalid step of Nº of nodes, it should be at least equal to maximum - minimum"+
                ", please check file: " + f)
    
    if hyperparameters["dense_dropout"][0] < 0 or hyperparameters["dense_dropout"][1] > 1:
        raise ValueError("Invalid dense_dropout value, the dense_dropout should be in the interval ]0, 1["+
        ", please check file: " + f)

    if hyperparameters["dense_dropout"][1] - hyperparameters["dense_dropout"][0] < hyperparameters["dense_dropout"][2] :
                raise ValueError("Invalid step of dropout, it should be at least equal to maximum - minimum"+
                ", please check file: " + f)
                  
    if hyperparameters['model']  == "cnn": # this will be changed in the future
        print("hello")        
    elif hyperparameters['model']  == "rnn":# this will be changed in the future
        print("hello")            
    elif  hyperparameters['model']  != "dense":
        raise ValueError(f"Invalid Neural network type - {hyperparameters['model']}, it should be one of the supported values [dense, cnn, rnn]"+
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

    output = f"./results/{dataset}"

    check_hyperparameters(params, f)

    if params["model"] == "dense":
        params["model"] = models.DenseModel
    elif params["model"] == "cnn":
        params["model"] = models.CNNModel
    else:
        params["model"] = models.RNNModel

    if dataset in jobs:
        jobs[dataset]["hyper_sets"].append(params) 

    else:
        jobs[dataset] = {"hyper_sets" : [params], "output" : output}

'''
    Load the dataset and run experiment
'''
for dataset in jobs:

    if dataset.lower() == "cifar10":
        train_data, test_data = dataset_loader.load_cifar10()
    elif dataset.lower() == "abalone":
        train_data, test_data = dataset_loader.load_abalone()
    elif dataset.lower() == "bike_sharing":
        train_data, test_data = dataset_loader.load_bike_sharing()
    elif dataset.lower() == "higgs":
        train_data, test_data = dataset_loader.load_higgs()
    elif dataset.lower() == "delays_zurich":
        train_data, test_data = dataset_loader.load_delays_zurich()
    elif dataset.lower() == "compas":
        train_data, test_data = dataset_loader.load_compas()
    elif dataset.lower() == "covertype":
        train_data, test_data = dataset_loader.load_covertype()
    else:
        if not path.exists(dataset):
            raise ValueError("Dataset file does not exist.")
        train_data, test_data= dataset_loader.load_dataset_from_file(dataset)

    x_train, y_train = train_data
    
    '''
    Just to understand the input shape
    '''
    
    if len(x_train) > 50000:
        x_train, y_train = shuffle(x_train, y_train, random_state=42, n_samples=1000)
    
    print(x_train.shape)
    x_train = x_train[:100]
    y_train = y_train[:100]
    
    '''
        Results folder creation
    '''
    results_dir = jobs[dataset]["output"]
    if not path.exists(results_dir):
        makedirs(results_dir)

    '''
        Run search
    '''
    search(dataset, jobs[dataset]["hyper_sets"], x_train, y_train, test_data, jobs[dataset]["output"])
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


