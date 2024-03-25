import numpy as np
import pandas as pd 
from fanova import fANOVA
from sklearn.preprocessing import LabelEncoder
import pathlib

np.random.seed(42)
le = LabelEncoder()

datasets = {"All Datasets" : ["abalone", "bike_sharing", "compas", "covertype", "delays_zurich", "higgs", "cifar10", "mnist", "fashion_mnist", "cifar100"],
            "Dense Regression" : ["abalone", "bike_sharing", "delays_zurich"],
            "Dense Classification" : ["compas", "covertype", "higgs", "cifar10", "mnist", "fashion_mnist", "cifar100"],
            "CNN Classification" : ["cifar10", "mnist", "fashion_mnist", "cifar100"],
            "Dense" : ["abalone", "bike_sharing", "compas", "covertype", "delays_zurich", "higgs", "cifar10", "mnist", "fashion_mnist", "cifar100"]}

hyperparameters = {
    "All Datasets"  : ["optimizer", "learning_rate", "loss", "batch_size"],
    "Dense Regression" : ["optimizer", "learning_rate", "loss", "batch_size", "n_dense_nodes",
                           "dense_activation", "dense_dropout_rate", "n_dense_layers"],

    "Dense Classification" : ["optimizer", "learning_rate", "loss", "batch_size", "n_dense_nodes",
                               "dense_activation", "dense_dropout_rate", "n_dense_layers"],

    "CNN Classification" : ["optimizer", "learning_rate", "loss", "batch_size", "n_kernels", "kernel_size",
               "pool_size", "conv_activation", "conv_dropout_rate", "n_conv_layers", 
               "n_dense_nodes", "dense_activation", "dense_dropout_rate", "n_dense_layers"],

    "Dense" : ["optimizer", "learning_rate", "loss", "batch_size", "n_dense_nodes", 
               "dense_activation", "dense_dropout_rate", "n_dense_layers"]
    }

models = {"All Datasets" : ["DenseModel", "CNNModel"],
            "Dense Regression" : ["DenseModel"],
            "Dense Classification" : ["DenseModel"],
            "CNN Classification" : ["CNNModel"],
            "Dense" : ["DenseModel"]

}

hyper_to_idx_dense = {
    "optimizer" : 0,
    "learning_rate" : 1,
    "loss" : 2,
    "n_dense_nodes" : 3,
    "dense_activation" : 4,
    "dense_dropout_rate" : 5,
    "n_dense_layers" : 6,
    "batch_size" : 7
}

hyper_to_idx_cnn = {
    "optimizer" : 0,
    "learning_rate" : 1,
    "loss" : 2,
    "n_kernels" : 3,
    "kernel_size" : 4,
    "pool_size" : 5,
    "conv_activation" : 6,
    "conv_dropout_rate" : 7,
    "n_conv_layers" : 8,
    "n_dense_nodes" : 9,
    "dense_activation" : 10,
    "dense_dropout_rate" : 11,
    "n_dense_layers" : 12,
    "batch_size" : 13
}

metrics = ["Performance", "Training time", "Inference time"]
fanova_results = {}

for dataset in datasets["All Datasets"]:
    dataset_folder = pathlib.Path(f"results/{dataset}")
    fanova_results[dataset] = {}

    for model in dataset_folder.glob("*"):
        if model.is_dir(): 
            model_name = str(model).split("/")[-1]
            results = pd.read_csv(model/"compiled_results.csv", header=0)
            results.dropna(inplace=True)
            results["optimizer"] = le.fit_transform(results["optimizer"])
            results["batch_size"] = le.fit_transform(results["batch_size"])
            results["loss"] = le.fit_transform(results["loss"])
            results["optimizer"] = le.fit_transform(results["optimizer"])
            results["dense_activation"] = le.fit_transform(results["dense_activation"])
            if model_name == "CNNModel":
                results["conv_activation"] = le.fit_transform(results["conv_activation"])

            #results["learning_rate"] = le.fit_transform(results["learning_rate"])

            if "mcc" in list(results.columns):
                results.drop(columns=["f1-score", "acc"], inplace=True)
            else:
                results.drop(columns=["mae"], inplace=True)
                
            X = results.values[:,: len(results.columns)-3]
            y_values = results.values[:,len(results.columns)-3:]

            fanova_classification = []                
            for idx in range(3):
                Y = y_values[:, idx]
                fanova_classification.append(fANOVA(X,Y))
            fanova_results[dataset][model_name] = fanova_classification

for test in hyperparameters:
    results = {}
    hyper_set = hyperparameters[test]
    model_count= 0
    for dataset in datasets[test]:
        #print(f"| | {dataset.capitalize()} | | |")
        #print(f"| Hyperparameter | Performance | Training Time | Inference Time |")
        #print("|---|---|---| ---- |")
        for model in fanova_results[dataset]:
            if model in models[test]:
                for hyperparameter in hyper_set:
                    if hyperparameter not in results:
                        results[hyperparameter] = [0,0,0]
        
                    idx = hyper_to_idx_cnn[hyperparameter] if "CNN" in model else  hyper_to_idx_dense[hyperparameter]
                    for i in range(3):
                        #print(idx, fanova_results[dataset][model][i].quantify_importance((idx,))[(idx,)]['individual importance'])
                        results[hyperparameter][i] += fanova_results[dataset][model][i].quantify_importance((idx,))[(idx,)]['individual importance']
                model_count +=1
    print(f"| | **{test}** | | |")
    print("|---|---|---| ---- |")
    print("| Hyperparameter | Performance | Training Time | Inference Time |")
    for hyperparamenter in hyper_set:
        row = f"| {hyperparamenter} | "
        for i in range(3):
            row += f"{(results[hyperparamenter][i]/model_count)*100:.2f} | "
        print(row)

    print("\n \n")

for dataset in fanova_results:
    for model in fanova_results[dataset]:
        print(f"| | **{dataset} {model}** | | |")
        print("|---|---|---| ---- |")
        print("| Hyperparameter | Performance | Training Time | Inference Time |")
        hyperparameters = hyper_to_idx_cnn if model == "CNNModel" else hyper_to_idx_dense

        for h, idx in hyperparameters.items():
            row = f"| {h} | "
            for i in range(3):
                res = fanova_results[dataset][model][i].quantify_importance((idx,))[(idx,)]['individual importance']
                row += f"{res*100:.2f} | "
            print(row)

        print("\n \n")