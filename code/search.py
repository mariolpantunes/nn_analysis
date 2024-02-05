
import numpy as np
import pandas as pd
import keras_tuner


def search(dataset_name, hyper_sets, x_train, y_train, val_data, output):
    results = {}
    
    ##Add here code to run the experiments
    for hyper_set in hyper_sets:
        model = hyper_set["model"](hyper_set)
        
        objective = keras_tuner.Objective("mcc", "max") if model.is_categorical else keras_tuner.Objective("mse", "min")

        tuner = keras_tuner.RandomSearch(
            model,
            max_trials=hyper_set["n_searches"],
            objective=objective,
            executions_per_trial=3,
            overwrite=True,
            directory=output,
            project_name=f"{type(hyper_set['model']).__name__}",
            seed=42
        )

        tuner.search_space_summary()

        ##Add here general dataset preprocessing (transform to greyscale etc?)

        tuner.search(
            x=x_train,
            y=y_train,
            validation_data=val_data,
        )

        tuner.results_summary()

        ##Add here code to obtain the metrics
        #param_names = []
        #for param_set in grid_result.cv_results_['params']:
        #    for key in param_set:    
        #        param_names.append(key)

    #param_names = list(set(param_names))
#
    #param_names = sorted(param_names)
#
    #for param_name in param_names:
    #    if param_name == "classifier__optimizer":
    #        values = [[],[]]
    #    else:
    #        values = []
    #    for param_set in grid_result.cv_results_['params']:
    #            if param_name in param_set:
    #                if param_name != "classifier":
    #                    if param_name == "classifier__optimizer":
    #                        values[0].append(param_set[param_name]._name)
    #                        values[1].append(param_set[param_name].learning_rate.numpy())
    #                    else:
    #                        values.append(param_set[param_name])
    #                else:
    #                    if "n_kernels" in param_set[param_name].get_params():
    #                        values.append("CNN")
    #                    elif "rnn_node" in param_set[param_name].get_params():
    #                        values.append("RNN")
    #                    else:
    #                        values.append("MLP")
    #            else:
    #                values.append(np.NaN)
#
    #    if param_name != "classifier":
    #        if param_name == "classifier__optimizer":
    #            results["optimizer"] = values[0]
    #            results["learning_rate"] = values[1]
    #        else:
    #            results[param_name.split("__")[1]] = values
    #    else: 
    #        results[param_name] = values
#
    #results['Performance (Avg)'] = np.round(grid_result.cv_results_['mean_test_score'],3)
    #results['Performance (Std)'] = np.round(grid_result.cv_results_['std_test_score'], 3)
    #results['Training Time (Avg)'] = np.round(grid_result.cv_results_['mean_fit_time'], 3)
    #results['Training Time (Std)'] = np.round(grid_result.cv_results_['std_fit_time'], 3)
    #results['Prediction Time (Avg)'] = np.round(grid_result.cv_results_['mean_score_time'], 3)
    #results['Prediction Time (Std)'] = np.round(grid_result.cv_results_['std_score_time'], 3)
    #results['dataset'] = dataset_name
#
    #df = pd.DataFrame(results)
#
    #df.to_csv(output_file, index=None)

