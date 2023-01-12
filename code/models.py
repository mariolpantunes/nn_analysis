from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow.keras.layers import (GRU, LSTM, BatchNormalization, Conv2D,
                                     Dense, Flatten, MaxPooling2D)
from tensorflow.keras.models import Sequential


def create_dense_model(classification):

    def model(input_shape, hidden_layer_dims, activation_functions, task_activation, task_nodes):
        model = Sequential()

        model.add(Dense(units=hidden_layer_dims[0], input_shape=input_shape, activation=activation_functions[0]))
        
        for i in range(1, len(hidden_layer_dims)):
            model.add(Dense(units=hidden_layer_dims[i], activation=activation_functions[i]))
        
        model.add(Dense(units=task_nodes,activation=task_activation))
        
        return model

    if classification:
        return KerasClassifier(model=model, 
                            verbose=0, 
                            input_shape=(768,),
                            hidden_layer_dim=[100],
                            activation_functions = ['relu'],
                            task_activation = 'softmax',
                            task_nodes = 1,
                            optimizer='adam',
                            loss='mean_squared_error'
                            )

    return KerasRegressor(model=model, 
                            verbose=0, 
                            input_shape=(768,),
                            hidden_layer_dims=[100],
                            activation_functions = ['relu'],
                            task_activation = 'linear',
                            task_nodes = 1,
                            optimizer='adam',
                            loss='mean_squared_error'
                            )


def create_cnn_model(classifier):

    def model(input_shape, n_kernels, kernel_sizes, pool_sizes, activation_functions, dense_sizes, dense_activations, task_activation, task_nodes):
        model = Sequential()
        model.add(Conv2D(n_kernels[0], kernel_size=kernel_sizes[0], activation=activation_functions[0], input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=pool_sizes[0]))
        model.add(BatchNormalization())

        for i in range(1, len(kernel_sizes)):
            model.add(Conv2D(n_kernels[i], kernel_size=kernel_sizes[i], activation=activation_functions[i]))
            model.add(MaxPooling2D(pool_size=pool_sizes[i]))
            model.add(BatchNormalization())

        model.add(Flatten())
        for i in range(len(dense_sizes)):
            model.add(Dense(dense_sizes[i], activation=dense_activations[i]))

        model.add(Dense(task_nodes, activation=task_activation))

        return model
    
    if classifier:
        return KerasClassifier(model=model, 
                            verbose=0, 
                            input_shape=(32,32,),
                            n_kernels=[32],
                            kernel_sizes=[(3,3)],
                            pool_sizes = [2],
                            activation_functions = ["relu"],
                            dense_sizes = [100],
                            dense_activations = ["relu"],
                            task_activation = "softmax",
                            task_nodes = 1
                            )

    return KerasRegressor(model=model, 
                            verbose=0, 
                            input_shape=(32,32,),
                            n_kernels=[32],
                            kernel_sizes=[(3,3)],
                            pool_sizes = [2],
                            activation_functions = ["relu"],
                            dense_sizes = [100],
                            dense_activations = ["relu"],
                            task_activation = "linear",
                            task_nodes = 1
                            )

def create_rnn_model(classifier):

    def model(input_shape, rnn_node, hidden_layer_dims, activation_functions, dense_sizes, dense_activations, task_activation, task_nodes):

        return_sequences = True if len(hidden_layer_dims) > 1 else False

        model = Sequential()
        rnn_node = GRU if rnn_node == "GRU" else LSTM
        model.add(rnn_node(units=hidden_layer_dims[0], input_shape=input_shape, activation=activation_functions[0]))

        for i in range(1, len(hidden_layer_dims)):
            model.add(rnn_node(units=hidden_layer_dims[i], activation=activation_functions[i], return_sequences=return_sequences))
            return_sequences = True if i < len(hidden_layer_dims) - 2 else False
        
        for i in range(len(dense_sizes)):
            model.add(Dense(units=dense_sizes[i],activation=dense_activations[i]))

        model.add(Dense(units=task_nodes,activation=task_activation))

        return model

    if classifier:
        return KerasClassifier(model=model, 
                            verbose=0, 
                            input_shape=(10,10),
                            rnn_node="GRU",
                            hidden_layer_dims=[10],
                            activation_functions = ["tanh"],
                            dense_sizes = [100],
                            dense_activations = ["relu"],
                            task_activation = 'softmax',
                            task_nodes = 1
                            )

    return KerasRegressor(model=model, 
                            verbose=0, 
                            input_shape=(10,10),
                            rnn_node="GRU",
                            hidden_layer_dims=[10],
                            activation_functions = ["tanh"],
                            dense_sizes = [100],
                            dense_activations = ["relu"],
                            task_activation = 'linear',
                            task_nodes = 1
                            )