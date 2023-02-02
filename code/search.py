
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def temporary_save(file):

    print(file)

def search(params, x_train, y_train, scorer, output_file, dataset_name):
    results = {}
    
    pipeline = Pipeline([('classifier', params[0]['classifier'])]) # simple initialization, 
                                                                    # the searchs runs the other classifiers

    grid = GridSearchCV(pipeline, params, n_jobs=1, cv=5, scoring=scorer, verbose=50, error_score=0)

    grid_result = grid.fit(x_train, y_train)#, callbacks=[temporary_save(output_file)])

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
    results['dataset'] = dataset_name

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(output_file)

