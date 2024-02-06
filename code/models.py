import keras_tuner
import keras
import numpy as np
from sklearn.metrics import matthews_corrcoef, mean_squared_error, accuracy_score, f1_score
from skimage.color import rgb2gray

def get_optimizer(optimizer):

    if optimizer.lower() == "adam":
        return keras.optimizers.Adam
    elif optimizer.lower() == "rmsprop":
        return keras.optimizers.RMSprop
    elif optimizer.lower() == "sgd":
        return keras.optimizers.SGD
    elif optimizer.lower() == "adadelta":
        return keras.optimizers.Adadelta
    elif optimizer.lower() == "adagrad":
        return keras.optimizers.Adagrad
    elif optimizer.lower() == "adamax":
        return keras.optimizers.Adamax
    elif optimizer.lower() == "nadam":
        return keras.optimizers.Nadam
    elif optimizer.lower() == "ftrl":
        return keras.optimizers.Ftrl
    else:
        raise ValueError(f'The optimizer {optimizer} is not supported!')

class DenseModel(keras_tuner.HyperModel):

    def __init__(self, hyperparameters):
        super().__init__()
        self.is_categorical = False
        self.hyperparameters = hyperparameters
        if "binary_crossentropy" in hyperparameters["loss"] or \
            "sparse_categorical_crossentropy" in hyperparameters["loss"] or \
            "categorical_crossentropy" in hyperparameters["loss"]:
            self.is_categorical = True
        
        self.metrics = ['accuracy'] if self.is_categorical else ['mean_squared_error']# Add new_metrics


    def build(self, hp):
        inputs = keras.Input(shape=(self.hyperparameters["input_shape"]))
        outputs = None

        optimizer = hp.Choice("optimizer", self.hyperparameters["optimizer"])
        learning_rate = hp.Float("learning_rate",  min_value=self.hyperparameters["lr"][0],
                                  max_value=self.hyperparameters["lr"][1], 
                                  step=self.hyperparameters["lr"][2])
        
        self.loss = hp.Choice("loss", self.hyperparameters["loss"])
        
        n_nodes = hp.Int(f"n_dense_nodes",  min_value=self.hyperparameters["n_dense_nodes"][0], 
                            max_value=self.hyperparameters["n_dense_nodes"][1], 
                            step=self.hyperparameters["n_dense_nodes"][2])
        
        activation = hp.Choice(f"dense_activation", self.hyperparameters["dense_activation_fn"])

        dropout_rate = hp.Float(f"dense_dropout_rate", min_value=self.hyperparameters["dense_dropout"][0], 
                                max_value=self.hyperparameters["dense_dropout"][1], 
                                step=self.hyperparameters["dense_dropout"][2])
            
        for i in range(hp.Int("n_dense_layers", min_value=self.hyperparameters["n_dense_layers"][0], 
                              max_value=self.hyperparameters["n_dense_layers"][1], 
                              step=self.hyperparameters["n_dense_layers"][2])):
            if outputs == None:
                outputs = keras.layers.Dense(n_nodes, activation=activation)(inputs)
            else:
                outputs = keras.layers.Dense(n_nodes, activation=activation)(outputs)

            outputs = keras.layers.Dropout(rate=dropout_rate)(outputs)
        

        if self.loss == "binary_crossentropy":
            outputs = keras.layers.Dense(units=1, activation="sigmoid")(outputs)
        elif self.loss in "sparse_categorical_crossentropy":
            outputs = keras.layers.Dense(units=self.hyperparameters["n_classes"], activation="sigmoid")(outputs)
        else:
            outputs = keras.layers.Dense(units=1, activation="linear")(outputs)
            self.is_categorical = False

        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=get_optimizer(optimizer)(learning_rate=learning_rate),
            loss=self.loss,
            metrics=self.metrics, 
        )
        return model

    def preprocess_data(self, x, y, validation_data):
        ##Add transformations based on the dataset we use
        ##Flatten images, remove time dependencies etc.

        if self.loss == "categorical_crossentropy":
            y = keras.utils.to_categorical(y)
            if validation_data:
                x_val, y_val = validation_data
                y_val = keras.utils.to_categorical(y_val)
                validation_data = (x_val, y_val)

        if self.hyperparameters["dataset"] == "cifar10":
            x_new = np.zeros((x.shape[0], x.shape[1]*x.shape[2]))

            for i in range(len(x)):
                x_new[i, :] = rgb2gray(x[i,:]).flatten()
            
            x = x_new

            if validation_data:
                x_val, y_val = validation_data
                x_val_new = np.zeros((x_val.shape[0], x_val.shape[1]*x_val.shape[2]))

                for i in range(len(x_val)):
                    x_val_new[i, :] = rgb2gray(x_val[i,:]).flatten()

                validation_data = (x_val_new, y_val)

        return x, y, validation_data

    #In case we want to optimize anything of the training
    def fit(self, hp, model, x, y, validation_data=None, *args, **kwargs):

        #Transform y_train and y_val in categorical and preprocess X to the correct format
        x, y, validation_data = self.preprocess_data(x, y, validation_data)

        batch_size = hp.Int("batch_size", min_value=self.hyperparameters["batch_size"][0], 
                                        max_value=self.hyperparameters["batch_size"][1], 
                                        step=self.hyperparameters["batch_size"][2])
        
        kwargs["callbacks"].append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=30))

        model.fit(
            x,
            y,
            batch_size= batch_size,
            epochs = self.hyperparameters["epochs"],
            validation_data=validation_data,
            verbose = 0,
            shuffle=True,
            **kwargs
        )

        if self.is_categorical:
            predictions = [round(x[0]) for x in model.predict(validation_data[0], verbose=0)] if self.loss == "binary_crossentropy" \
                            else [np.argmax(x) for x in model.predict(validation_data[0], verbose=0)]
            
            y_val =  [np.argmax(x) for x in validation_data[1]] if self.loss == "categorical_crossentropy" else validation_data[1]

            return {'mcc' : matthews_corrcoef(y_val, predictions), 'acc' : accuracy_score(y_val, predictions), 'f1' : f1_score(y_val, predictions, average="macro")}
        else:
            predictions = model.predict(validation_data[0], verbose=0).reshape((-1,))
            results = mean_squared_error(validation_data[1], predictions)
            return {'mse' : results}

class CNNModel(keras_tuner.HyperModel):

    def __init__(self, hyperparameters):
        super().__init__()

        self.is_categorical = False
        self.hyperparameters = hyperparameters
        if "binary_crossentropy" in hyperparameters["loss"] or \
            "sparse_categorical_crossentropy" in hyperparameters["loss"] or \
            "categorical_crossentropy" in hyperparameters["loss"]:
            self.is_categorical = True
        
        self.metrics = ['accuracy'] if self.is_categorical else ['mean_squared_error']# Add new_metrics


    def build(self, hp):

        if self.hyperparameters["dimension"] == 1:
            conv_layer = keras.layers.Conv1D
            maxpooling_layer = keras.layers.MaxPooling1D

        elif self.hyperparameters["dimension"] == 2:
            conv_layer = keras.layers.Conv2D
            maxpooling_layer = keras.layers.MaxPooling2D

        else:
            conv_layer = keras.layers.Conv3D
            maxpooling_layer = keras.layers.MaxPooling3D

        inputs = keras.Input(shape=(self.hyperparameters["input_shape"]))
        outputs = None

        optimizer = hp.Choice("optimizer", self.hyperparameters["optimizer"])
        learning_rate = hp.Float("learning_rate",  min_value=self.hyperparameters["lr"][0],
                                  max_value=self.hyperparameters["lr"][1], 
                                  step=self.hyperparameters["lr"][2])
        
        self.loss = hp.Choice("loss", self.hyperparameters["loss"])
        
        n_kernels = hp.Int(f"n_kernels",  min_value=self.hyperparameters["n_kernels"][0], 
                            max_value=self.hyperparameters["n_kernels"][1], 
                            step=self.hyperparameters["n_kernels"][2])
        
        kernel_size = hp.Int(f"kernel_size",  min_value=self.hyperparameters["kernel_size"][0], 
                            max_value=self.hyperparameters["kernel_size"][1], 
                            step=self.hyperparameters["kernel_size"][2])
        
        pool_size = hp.Int(f"pool_size",  min_value=self.hyperparameters["pool_size"][0], 
                            max_value=self.hyperparameters["pool_size"][1], 
                            step=self.hyperparameters["pool_size"][2])
        
        activation = hp.Choice(f"conv_activation", self.hyperparameters["conv_activation_fn"])

        dropout_rate = hp.Float(f"conv_dropout_rate", min_value=self.hyperparameters["conv_dropout"][0], 
                                max_value=self.hyperparameters["conv_dropout"][1], 
                                step=self.hyperparameters["conv_dropout"][2])
        
        for i in range(hp.Int("n_conv_layers", min_value=self.hyperparameters["n_conv_layers"][0], 
                              max_value=self.hyperparameters["n_conv_layers"][1], 
                              step=self.hyperparameters["n_conv_layers"][2])):
            
            if outputs == None:
                outputs = conv_layer(n_kernels, 
                    kernel_size=kernel_size, 
                    activation=activation)(inputs)
            else:
                outputs = conv_layer(n_kernels, 
                    kernel_size=kernel_size, 
                    activation=activation)(outputs)
                
        outputs = maxpooling_layer(pool_size=pool_size)(outputs)
        outputs = keras.layers.BatchNormalization()(outputs)
        outputs = keras.layers.Dropout(rate=dropout_rate)(outputs)

        outputs = keras.layers.Flatten()(outputs)
        
        n_nodes = hp.Int(f"n_dense_nodes",  min_value=self.hyperparameters["n_dense_nodes"][0], 
                            max_value=self.hyperparameters["n_dense_nodes"][1], 
                            step=self.hyperparameters["n_dense_nodes"][2])
        
        activation = hp.Choice(f"dense_activation", self.hyperparameters["dense_activation_fn"])

        dropout_rate = hp.Float(f"dense_dropout_rate", min_value=self.hyperparameters["dense_dropout"][0], 
                                max_value=self.hyperparameters["dense_dropout"][1], 
                                step=self.hyperparameters["dense_dropout"][2])
            
        for i in range(hp.Int("n_dense_layers", min_value=self.hyperparameters["n_dense_layers"][0], 
                              max_value=self.hyperparameters["n_dense_layers"][1], 
                              step=self.hyperparameters["n_dense_layers"][2])):
            
            outputs = keras.layers.Dense(n_nodes, activation=activation)(outputs)

            outputs = keras.layers.Dropout(rate=dropout_rate)(outputs)
        
        if self.loss == "binary_crossentropy":
            outputs = keras.layers.Dense(units=1, activation="sigmoid")(outputs)
        elif self.loss in "sparse_categorical_crossentropy":
            outputs = keras.layers.Dense(units=self.hyperparameters["n_classes"], activation="sigmoid")(outputs)
        else:
            outputs = keras.layers.Dense(units=1, activation="linear")(outputs)
            self.is_categorical = False

        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=get_optimizer(optimizer)(learning_rate=learning_rate),
            loss=self.loss,
            metrics=self.metrics, 
        )
        return model

    def preprocess_data(self, x, y, validation_data):
        ##Add transformations based on the dataset we use
        ##Flatten images, remove time dependencies etc.

        if self.loss == "categorical_crossentropy":
            y = keras.utils.to_categorical(y)
            if validation_data:
                x_val, y_val = validation_data
                y_val = keras.utils.to_categorical(y_val)
                validation_data = (x_val, y_val)

        return x, y, validation_data
    
    #In case we want to optimize anything of the training
    def fit(self, hp, model, x, y, validation_data=None, **kwargs):

        #Transform y_train and y_val in categorical and preprocess X to the correct format
        x, y, validation_data = self.preprocess_data(x, y, validation_data)

        batch_size = hp.Int("batch_size", min_value=self.hyperparameters["batch_size"][0], 
                                        max_value=self.hyperparameters["batch_size"][1], 
                                        step=self.hyperparameters["batch_size"][2])
        
        kwargs["callbacks"].append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=30))

        model.fit(
            x,
            y,
            batch_size= batch_size,
            epochs = self.hyperparameters["epochs"],
            validation_data=validation_data,
            verbose = 0,
            shuffle=True,
            **kwargs
        )

        if self.is_categorical:
            predictions = [round(x[0]) for x in model.predict(validation_data[0], verbose=0)] if self.loss == "binary_crossentropy" \
                            else [np.argmax(x) for x in model.predict(validation_data[0], verbose=0)]
            
            y_val =  [np.argmax(x) for x in validation_data[1]] if self.loss == "categorical_crossentropy" else validation_data[1]

            return {'mcc' : matthews_corrcoef(y_val, predictions), 'acc' : accuracy_score(y_val, predictions), 'f1' : f1_score(y_val, predictions, average="macro")}
        else:
            predictions = model.predict(validation_data[0], verbose=0).reshape((-1,))
            results = mean_squared_error(validation_data[1], predictions)
            return {'mse' : results}
    
class RNNModel(keras_tuner.HyperModel):

    def __init__(self, hyperparameters):
        super().__init__()
        self.is_categorical = False
        self.hyperparameters = hyperparameters
        if "binary_crossentropy" in hyperparameters["loss"] or \
            "sparse_categorical_crossentropy" in hyperparameters["loss"] or \
            "categorical_crossentropy" in hyperparameters["loss"]:
            self.is_categorical = True
        
        self.metrics = ['accuracy'] if self.is_categorical else ['mean_squared_error']# Add new_metrics


    def build(self, hp):
        inputs = keras.Input(shape=(self.hyperparameters["input_shape"]))
        outputs = None
        rnn_layer = hp.Choice("rnn_layer", self.hyperparameters["rnn_layer"])
        if rnn_layer == "lstm":
            rnn_layer = keras.layers.LSTM
        elif rnn_layer == "gru":
            rnn_layer = keras.layers.GRU


        optimizer = hp.Choice("optimizer", self.hyperparameters["optimizer"])
        learning_rate = hp.Float("learning_rate",  min_value=self.hyperparameters["lr"][0],
                                  max_value=self.hyperparameters["lr"][1], 
                                  step=self.hyperparameters["lr"][2])
        
        self.loss = hp.Choice("loss", self.hyperparameters["loss"])


        n_nodes = hp.Int(f"n_rnn_nodes",  min_value=self.hyperparameters["n_rnn_nodes"][0], 
                            max_value=self.hyperparameters["n_rnn_nodes"][1], 
                            step=self.hyperparameters["n_rnn_nodes"][2])
        
        activation = hp.Choice(f"rnn_activation", self.hyperparameters["rnn_activation"])

        dropout_rate = hp.Float(f"rnn_dropout_rate", min_value=self.hyperparameters["rnn_dropout"][0], 
                                max_value=self.hyperparameters["rnn_dropout"][1], 
                                step=self.hyperparameters["rnn_dropout"][2])
        n_layers = hp.Int("n_rnn_layers", min_value=self.hyperparameters["n_rnn_layers"][0], 
                              max_value=self.hyperparameters["n_rnn_layers"][1], 
                              step=self.hyperparameters["n_rnn_layers"][2])
        for i in range():
            return_sequences = True if i < len(n_layers) -1 else False
            if outputs == None:
                outputs = rnn_layer(units=n_nodes, 
                    activation=activation, 
                    return_sequences=return_sequences)(inputs)
            else:
                outputs = rnn_layer(units=n_nodes, 
                    activation=activation, 
                    return_sequences=return_sequences)(outputs)
                
            outputs = keras.layers.Dropout(rate=dropout_rate)(outputs)

        
        n_nodes = hp.Int(f"n_dense_nodes",  min_value=self.hyperparameters["n_dense_nodes"][0], 
                            max_value=self.hyperparameters["n_dense_nodes"][1], 
                            step=self.hyperparameters["n_dense_nodes"][2])
        
        activation = hp.Choice(f"dense_activation", self.hyperparameters["dense_activation_fn"])

        dropout_rate = hp.Float(f"dense_dropout_rate", min_value=self.hyperparameters["dense_dropout"][0], 
                                max_value=self.hyperparameters["dense_dropout"][1], 
                                step=self.hyperparameters["dense_dropout"][2])
            
        for i in range(hp.Int("n_dense_layers", min_value=self.hyperparameters["n_dense_layers"][0], 
                              max_value=self.hyperparameters["n_dense_layers"][1], 
                              step=self.hyperparameters["n_dense_layers"][2])):
            if outputs == None:
                outputs = keras.layers.Dense(n_nodes, activation=activation)(inputs)
            else:
                outputs = keras.layers.Dense(n_nodes, activation=activation)(outputs)

            outputs = keras.layers.Dropout(rate=dropout_rate)(outputs)
        

        if self.loss == "binary_crossentropy":
            outputs = keras.layers.Dense(units=1, activation="sigmoid")(outputs)
        elif self.loss in "sparse_categorical_crossentropy":
            outputs = keras.layers.Dense(units=self.hyperparameters["n_classes"], activation="sigmoid")(outputs)
        else:
            outputs = keras.layers.Dense(units=1, activation="linear")(outputs)
            self.is_categorical = False

        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=get_optimizer(optimizer)(learning_rate=learning_rate),
            loss=self.loss,
            metrics=self.metrics, 
        )
        return model

    def preprocess_data(self, x, y, validation_data):
        ##Add transformations based on the dataset we use
        ##Flatten images, remove time dependencies etc.

        if self.loss == "categorical_crossentropy":
            y = keras.utils.to_categorical(y)
            if validation_data:
                x_val, y_val = validation_data
                y_val = keras.utils.to_categorical(y_val)
                validation_data = (x_val, y_val)

        if self.hyperparameters["dataset"] == "cifar10":
            x_new = np.zeros((x.shape[0], x.shape[1]*x.shape[2]))

            for i in range(len(x)):
                x_new[i, :] = rgb2gray(x[i,:]).flatten()
            
            x = x_new

            if validation_data:
                x_val, y_val = validation_data
                x_val_new = np.zeros((x_val.shape[0], x_val.shape[1]*x_val.shape[2]))

                for i in range(len(x_val)):
                    x_val_new[i, :] = rgb2gray(x_val[i,:]).flatten()

                validation_data = (x_val_new, y_val)

        return x, y, validation_data

    #In case we want to optimize anything of the training
    def fit(self, hp, model, x, y, validation_data=None, *args, **kwargs):

        #Transform y_train and y_val in categorical and preprocess X to the correct format
        x, y, validation_data = self.preprocess_data(x, y, validation_data)

        batch_size = hp.Int("batch_size", min_value=self.hyperparameters["batch_size"][0], 
                                        max_value=self.hyperparameters["batch_size"][1], 
                                        step=self.hyperparameters["batch_size"][2])
        
        kwargs["callbacks"].append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=30))

        model.fit(
            x,
            y,
            batch_size= batch_size,
            epochs = self.hyperparameters["epochs"],
            validation_data=validation_data,
            verbose = 0,
            shuffle=True,
            **kwargs
        )

        if self.is_categorical:
            predictions = [round(x[0]) for x in model.predict(validation_data[0], verbose=0)] if self.loss == "binary_crossentropy" \
                            else [np.argmax(x) for x in model.predict(validation_data[0], verbose=0)]
            
            y_val =  [np.argmax(x) for x in validation_data[1]] if self.loss == "categorical_crossentropy" else validation_data[1]

            return {'mcc' : matthews_corrcoef(y_val, predictions), 'acc' : accuracy_score(y_val, predictions), 'f1' : f1_score(y_val, predictions, average="macro")}
        else:
            predictions = model.predict(validation_data[0], verbose=0).reshape((-1,))
            results = mean_squared_error(validation_data[1], predictions)
            return {'mse' : results}