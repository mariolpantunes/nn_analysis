
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np

def search(params, x_train, y_train, scorer, outputFolder):
    results = {}

    pipeline = Pipeline([('classifier', params[0]['classifier'])])

    grid = GridSearchCV(pipeline, params, n_jobs=1, cv=3, scoring=scorer, verbose=3)

    grid_result = grid.fit(x_train, y_train)

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
    df.to_csv(outputFolder+"test.csv")