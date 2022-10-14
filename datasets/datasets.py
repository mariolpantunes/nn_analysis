import pandas as pd
import numpy as np

from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler


#Helper function
def vectorize_sequences(sequences, dimension=10000):

    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  
    return results


def load_abalone():
    data = pd.read_csv('./datasets/abalone.csv')


    sex = data.pop('Sex')
 
    data['M'] = (sex == 'M')*1.0
    data['F'] = (sex == 'F')*1.0
    data['I'] = (sex == 'I')*1.0

     
    dataset = data[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight','Viscera weight','Shell weight','M','F','I','Rings']]
    

    X = dataset.iloc[:,0:10]
    y = dataset.iloc[:,10].values

    scalar= MinMaxScaler()
    X = scalar.fit_transform(X)
    y = y.reshape(-1,1)
    y = scalar.fit_transform(y)


    return X, y


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