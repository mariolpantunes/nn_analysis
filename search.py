
from tabnanny import verbose
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, matthews_corrcoef

from keras.utils import to_categorical
import pandas as pd
import numpy as np

import models
import datasets.datasets as datasets
import json

def mcc(y_pred, y_true):
    y_true =[np.argmax(x) for x in y_true]
    y_pred =[np.argmax(x) for x in y_pred]

    return matthews_corrcoef(y_true, y_pred)

def search(params, X_train, y_train, scorer):
    results = {}

    pipeline = Pipeline([('classifier', params[0]['classifier'])])

    grid = GridSearchCV(pipeline, params, n_jobs=1, cv=3, scoring=scorer, verbose=3)

    grid_result = grid.fit(X_train, y_train)

    param_names = []
    for param_set in grid_result.cv_results_['params']:
        for key in param_set:
            if key != 'classifier':
                param_names.append(key)

    param_names = set(param_names)

    for param_name in param_names:
        values = []
        for param_set in grid_result.cv_results_['params']:
                if param_name in param_set:
                    values.append(param_set[param_name])
                else:
                    values.append(np.NaN)

        results[param_name.split("__")[1]] = values

    results['Performance (Avg)'] = np.round(grid_result.cv_results_['mean_test_score'],3)
    results['Performance (Std)'] = np.round(grid_result.cv_results_['std_test_score'], 3)
    results['Training Time (Avg)'] = np.round(grid_result.cv_results_['mean_fit_time'], 3)
    results['Training Time (Std)'] = np.round(grid_result.cv_results_['std_fit_time'], 3)
    results['Prediction Time (Avg)'] = np.round(grid_result.cv_results_['mean_score_time'], 3)
    results['Training Time (Std)'] = np.round(grid_result.cv_results_['std_score_time'], 3)

    df = pd.DataFrame(results)
    df.to_csv("test.csv")
    print(df)

print("Creating Model")
dense_model = models.create_dense_model(True)
cnn_model = models.create_cnn_model(True)
gru_model = models.create_gru_model(True)

print("Loading Dataset")
train_data, test_data= datasets.load_cifar10()

x_train, y_train = train_data

y_train = to_categorical(y_train)

print("Hyperparameter Optmization")

#Training for dense model
x_train_dense = np.reshape(x_train, (x_train.shape[0], -1))
dense_model_param = json.load(open("hyperparameters/FashionMNIST/DENSE/dense_param_examples.json"))
dense_model_param["classifier"] = [dense_model]
dense_params = [dense_model_param]

search(dense_params, x_train_dense, y_train, make_scorer(mcc, greater_is_better=True))



#Training for cnn model
#cnn_model_param = json.load(open("hyperparameters/FashionMNIST/CNN/cnn_param_examples.json"))
#cnn_model_param["classifier"] = [cnn_model]
#cnn_params = [cnn_model_param]
#
#search(cnn_params, x_train, y_train, make_scorer(mcc, greater_is_better=True))

# Training for GRU model
#gru_model_param = json.load(open("hyperparameters/FashionMNIST/GRU/gru_param_examples.json"))
#gru_model_param["classifier"] = [gru_model]
#gru_params = [gru_model_param]
#x_train_gru = x_train[:,:,:,0]
#
#search(gru_params, x_train_gru, y_train, make_scorer(mcc, greater_is_better=True))
