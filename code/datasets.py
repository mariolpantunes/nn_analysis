import pandas as pd
from sklearn import datasets
from tensorflow.keras.datasets import cifar10, fashion_mnist


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Parse numbers as floats
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize data
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)


def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize data
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)


def load_dataset(file_location): # this needs to be changed as well as the base abalone.csv file
                                 # Change this to a standard .csv load and division into train and test
                                 # make a change in the abalone.csv to standerdize it so we only need to load, not preprocess.

    if file_location.split(".")[-1] != "csv":
        raise ValueError("Dataset should be a csv file.")

    data = pd.read_csv(file_location, index_col=False)

    train_length = int(data.shape[0]*0.8)

    x_train = data.iloc[:train_length,:data.shape[1]-1].values
    y_train = data.iloc[:train_length, data.shape[1]-1].values

    x_test = data.iloc[train_length: , :data.shape[1]-1].values
    y_test = data.iloc[train_length: , data.shape[1]-1].values

    return (x_train, y_train), (x_test, y_test)