
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def search(params, x_train, y_train, scorer, output_file, dataset_name):
    results = {}
    
    pipeline = Pipeline([('classifier', params[0]['classifier'])]) # simple initialization, 
                                                                    # the searchs runs the other classifiers

    grid = GridSearchCV(pipeline, params, n_jobs=1, cv=5, scoring=scorer, verbose=50, error_score=0)

    grid_result = grid.fit(x_train, y_train)#, callbacks=[temporary_save(output_file)])

    param_names = []
    for param_set in grid_result.cv_results_['params']:
        for key in param_set:    
            param_names.append(key)

    param_names = list(set(param_names))

    param_names = sorted(param_names)

    for param_name in param_names:
        if param_name == "classifier__optimizer":
            values = [[],[]]
        else:
            values = []
        for param_set in grid_result.cv_results_['params']:
                if param_name in param_set:
                    if param_name != "classifier":
                        if param_name == "classifier__optimizer":
                            values[0].append(param_set[param_name]._name)
                            values[1].append(param_set[param_name].learning_rate.numpy())
                        else:
                            values.append(param_set[param_name])
                    else:
                        if "n_kernels" in param_set[param_name].get_params():
                            values.append("CNN")
                        elif "rnn_node" in param_set[param_name].get_params():
                            values.append("RNN")
                        else:
                            values.append("MLP")
                else:
                    values.append(np.NaN)

        if param_name != "classifier":
            if param_name == "classifier__optimizer":
                results["optimizer"] = values[0]
                results["learning_rate"] = values[1]
            else:
                results[param_name.split("__")[1]] = values
        else: 
            results[param_name] = values

    results['Performance (Avg)'] = np.round(grid_result.cv_results_['mean_test_score'],3)
    results['Performance (Std)'] = np.round(grid_result.cv_results_['std_test_score'], 3)
    results['Training Time (Avg)'] = np.round(grid_result.cv_results_['mean_fit_time'], 3)
    results['Training Time (Std)'] = np.round(grid_result.cv_results_['std_fit_time'], 3)
    results['Prediction Time (Avg)'] = np.round(grid_result.cv_results_['mean_score_time'], 3)
    results['Training Time (Std)'] = np.round(grid_result.cv_results_['std_score_time'], 3)
    results['dataset'] = dataset_name

    df = pd.DataFrame(results)

    df.to_csv(output_file, index=None)

