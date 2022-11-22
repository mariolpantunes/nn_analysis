from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GRU
from scikeras.wrappers import KerasClassifier, KerasRegressor

def create_dense_model(classification):

    def model(input_shape, hidden_layer_dim, n_hidden_layers, activation_function, task_activation, task_nodes):
        model = Sequential()
        model.add(Dense(units=hidden_layer_dim, input_shape=input_shape, activation=activation_function))
        for i in range(n_hidden_layers -1):
            model.add(Dense(units=hidden_layer_dim, activation=activation_function))
        
        model.add(Dense(units=task_nodes,activation=task_activation))
        
        return model

    if classification:
        return KerasClassifier(model=model, 
                            verbose=0, 
                            input_shape=(768,),
                            hidden_layer_dim=10,
                            n_hidden_layers = 1,
                            activation_function = 'relu',
                            task_activation = 'softmax',
                            task_nodes = 1,
                            optimizer='adam',
                            loss='mean_squared_error'
                            )

    return KerasRegressor(model=model, 
                            verbose=0, 
                            input_shape=(768,),
                            hidden_layer_dim=10,
                            n_hidden_layers = 1,
                            activation_function = 'relu',
                            task_activation = 'linear',
                            task_nodes = 1,
                            optimizer='adam',
                            loss='mean_squared_error'
                            )


def create_cnn_model(classifier):

    def model(input_shape, n_kernels, kernel_size, pool_size, activation_function, n_hidden_layers, task_activation, task_nodes):
        model = Sequential()
        model.add(Conv2D(n_kernels, kernel_size=kernel_size, activation=activation_function, input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(BatchNormalization())
        for i in range(n_hidden_layers-1):
            model.add(Conv2D(n_kernels, kernel_size=kernel_size, activation=activation_function))
            model.add(MaxPooling2D(pool_size=pool_size))
            model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(256, activation=activation_function))
        model.add(Dense(128, activation=activation_function))
        model.add(Dense(task_nodes, activation=task_activation))

        return model
    
    if classifier:
        return KerasClassifier(model=model, 
                            verbose=0, 
                            input_shape=(32,32,),
                            n_kernels=32,
                            kernel_size=(3,3),
                            pool_size = 2,
                            n_hidden_layers = 1,
                            activation_function = "relu",
                            task_activation = "softmax",
                            task_nodes = 1
                            )

    return KerasRegressor(model=model, 
                            verbose=0, 
                            input_shape=(32,32,),
                            n_kernels=32,
                            kernel_size=(3,3),
                            pool_size = 2,
                            n_hidden_layers = 1,
                            activation_function = "relu",
                            task_activation = "linear",
                            task_nodes = 1
                            )

def create_gru_model(classifier):

    def model(input_shape, hidden_layer_dim, n_hidden_layers, activation_function, task_activation, task_nodes):

        return_sequences = True if n_hidden_layers > 1 else False

        model = Sequential()
        model.add(GRU(units=hidden_layer_dim, input_shape=input_shape, activation=activation_function))
        for i in range(n_hidden_layers -1):
            model.add(GRU(units=hidden_layer_dim, activation=activation_function, return_sequences=return_sequences))
            return_sequences = True if n_hidden_layers > 1 else False
        
        model.add(Dense(units=task_nodes,activation=task_activation))

        return model

    if classifier:
        return KerasClassifier(model=model, 
                            verbose=0, 
                            input_shape=(10,10),
                            hidden_layer_dim=10,
                            n_hidden_layers = 1,
                            activation_function = "tanh",
                            task_activation = 'softmax',
                            task_nodes = 1,
                            loss= "categorical_crossentropy"
                            )

    return KerasRegressor(model=model, 
                            verbose=0, 
                            input_shape=(10,10),
                            hidden_layer_dim=10,
                            n_hidden_layers = 1,
                            activation_function = "tanh",
                            task_activation = 'linear',
                            task_nodes = 1
                            )