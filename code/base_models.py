import pathlib

import keras_tuner
import numpy as np
import tensorflow as tf
from dataset_loader import (load_abalone, load_bike_sharing, load_compas,
                            load_covertype, load_delays_zurich, load_higgs)
from sklearn.metrics import matthews_corrcoef, mean_squared_error
from sklearn.utils import shuffle
from tensorflow.keras.layers import (GRU, LSTM, BatchNormalization, Conv1D,
                                     Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling1D, MaxPooling2D)
from tensorflow.keras.models import Sequential

tf.keras.utils.set_random_seed(42)

def run_regression(output):
    datasets = ["bike_sharing", "delays_zurich", "abalone"]

    for dataset in datasets:
        if dataset.lower() == "abalone":
            train_data, test_data = load_abalone()
        elif dataset.lower() == "bike_sharing":
            train_data, test_data = load_bike_sharing()
        elif dataset.lower() == "delays_zurich":
            train_data, test_data = load_delays_zurich()

        output_folder = output / dataset
        output_folder.mkdir(parents=True, exist_ok=True)
        x_train, y_train = train_data

        if len(x_train) > 100000:
            x_train, y_train = shuffle(x_train, y_train, random_state=42, n_samples=100000)

        def build_model(hp):
            model = Sequential()
            model.add(Dense(
                    hp.Int("units_0", min_value=32, max_value=1024, step=32),
                    activation='relu', input_shape=x_train.shape[1:], name="dense_0"))
            if hp.Boolean("dropout_0"):
                model.add(Dropout(rate=hp.Float("dropout_0_value", min_value=0.01, max_value=0.5, step=0.0001)))
            for i in range(hp.Int("n_hidden_layers", min_value=3, max_value=7, step=1)):
                model.add(Dense(
                    hp.Int(f"units_{i+1}", min_value=32, max_value=1024, step=32),
                    activation='relu', input_shape=x_train.shape[1:], name=f"dense_{i+1}"))
                if hp.Boolean(f"dropout_{i+1}"):
                    model.add(Dropout(rate=hp.Float(f"dropout_{i+1}_value", min_value=0.01, max_value=0.5, step=0.0001)))

            model.add(Dense(1, activation='linear', name=f"dense_{i+2}"))
            model.compile(optimizer="adam", loss="mean_squared_error")
            return model

        tuner = keras_tuner.RandomSearch(
            hypermodel=build_model,
            objective="val_loss",
            max_trials=100,
            executions_per_trial=1,
            overwrite=True,
            directory=output_folder,
            project_name="random_search",
        )

        tuner.search(x_train, y_train, batch_size=1024, epochs=200, validation_split=0.2, verbose=0)

        best_hps = tuner.get_best_hyperparameters(1)
        model = build_model(best_hps[0])
        model.fit(x_train, y_train, batch_size=1024, epochs=300, validation_split=0.2, verbose=2)
        model.save(output_folder / "best_model")

        x_test, y_test = test_data

        y_pred = model.predict(x_test)
        y_pred = [y[0] for y in y_pred]

        print(mean_squared_error(y_pred, y_test))

def run_classification(output):
        datasets = ["compas", "covertype", "higgs"]
        n_features = 0
        for dataset in datasets:
            if dataset.lower() == "compas":
                train_data, test_data = load_compas()
                n_features = max(train_data[1])+1
            elif dataset.lower() == "covertype":
                train_data, test_data = load_covertype()
                n_features = max(train_data[1])+1

            elif dataset.lower() == "higgs":
                train_data, test_data = load_higgs()
                n_features = max(train_data[1])+1

            output_folder = output / dataset
            output_folder.mkdir(parents=True, exist_ok=True)
            x_train, y_train = train_data
            y_train = tf.keras.utils.to_categorical(y_train)

            if len(x_train) > 100000:
                x_train, y_train = shuffle(x_train, y_train, random_state=42, n_samples=50000)

            def build_model(hp):
                model = Sequential()
                model.add(Dense(
                        hp.Int("units_0", min_value=32, max_value=1024, step=32),
                        activation='relu', input_shape=x_train.shape[1:], name="dense_0"))
                if hp.Boolean("dropout_0"):
                    model.add(Dropout(rate=hp.Float("dropout_0_value", min_value=0.01, max_value=0.5, step=0.0001)))
                for i in range(hp.Int("n_hidden_layers", min_value=3, max_value=7, step=1)):
                    model.add(Dense(
                        hp.Int(f"units_{i+1}", min_value=32, max_value=1024, step=32),
                        activation='relu', input_shape=x_train.shape[1:], name=f"dense_{i+1}"))
                    if hp.Boolean(f"dropout_{i+1}"):
                        model.add(Dropout(rate=hp.Float(f"dropout_{i+1}_value", min_value=0.01, max_value=0.5, step=0.0001)))

                model.add(Dense(n_features, activation='softmax', name=f"dense_{i+2}"))
                model.compile(optimizer="adam", loss="categorical_crossentropy")
                return model

            tuner = keras_tuner.RandomSearch(
                hypermodel=build_model,
                objective="val_loss",
                max_trials=100,
                executions_per_trial=100,
                overwrite=True,
                directory=output_folder,
                project_name="random_search",
            )

            tuner.search(x_train, y_train, batch_size=1024, epochs=200, validation_split=0.2, verbose=0)

            best_hps = tuner.get_best_hyperparameters(1)
            model = build_model(best_hps[0])
            model.fit(x_train, y_train, batch_size=1024, epochs=300 , validation_split=0.2, verbose=2)
            model.save(output_folder / "best_model")

            x_test, y_test = test_data

            y_pred = model.predict(x_test)
            y_pred = [np.argmax(x) for x in y_pred]

            print(matthews_corrcoef(y_pred, y_test))

    
results_base_path = "../results/trial_3/"
output = pathlib.Path(results_base_path)
run_classification(output)