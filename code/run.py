import argparse
import gc
import glob
import json
from os import makedirs, path

import dataset_loader
import models
import tensorflow as tf
from metrics import mcc, mse
from search import search
from sklearn.utils import shuffle

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
    params["classifier"] = [get_model(params["model"], params["is_categorical"])]

    if "classifier__learning_rate" in params:
        optimizers = []
        for lr in params["classifier__learning_rate"]:
            for optimizer in params["classifier__optimizer"]:
                optimizers.append(models.get_optimizer(optimizer, lr))
        params["classifier__optimizer"] = optimizers
        del params["classifier__learning_rate"]

    else:
        params["classifier__optimizer"] = [models.get_optimizer(optimizer) for optimizer in params["classifier__optimizer"]]

    scorer = mcc() if params["is_categorical"] else "neg_root_mean_squared_error"
    output = params["output"]

    check_hyperparameters(params, f)

    del params["dataset"]
    del params["is_categorical"]
    del params["model"]
    del params["output"]

    if dataset in jobs:

        jobs[dataset]["hyper_set"].append(params) 

    else:

        jobs[dataset] = {"hyper_set" : [params], "scorer" : scorer, "output" : output}

    params["classifier__callbacks"] = [[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)]]
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
    elif dataset.lower() == "sms_spam_collection":
        train_data, test_data = dataset_loader.load_sms_spam_collection()
    else:
        if not path.exists(dataset):
            raise ValueError("Dataset file does not exist.")
        train_data, test_data= dataset_loader.load_dataset_from_file(dataset)

    x_train, y_train = train_data
    #x_train = x_train[:,:,:,0]
    if len(x_train) > 100000:
        x_train, y_train = shuffle(x_train, y_train, random_state=42, n_samples=50000)
    
    for hyper_set in jobs[dataset]["hyper_set"]:
        hyper_set["classifier__input_shape"] = [x_train.shape[1:]]

    if str(jobs[dataset]["scorer"]) == "make_scorer(mcc)" and jobs[dataset]["hyper_set"][0]["classifier__loss"][0] == "categorical_crossentropy":
        y_train = tf.keras.utils.to_categorical(y_train)

    '''
        Results folder creation
    '''
    results_dir = "/".join(jobs[dataset]["output"].split("/")[:-1])
    if not path.exists(results_dir):
        makedirs(results_dir)

    '''
        Run search
    '''
    search(jobs[dataset]["hyper_set"], x_train, y_train, jobs[dataset]["scorer"], jobs[dataset]["output"], dataset)
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


