import pandas as pd
import numpy as np

from tensorflow.keras.datasets import mnist, imdb, cifar10
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


def load_banknote():
    data = pd.read_csv('./datasets/banknote.csv')

    X, y = data.values[:, :-1], data.values[:, -1]

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
    image_shape = (28,28,1) 

    train_df = pd.read_csv('./datasets/fashion-mnist_train.csv', sep=',')
    test_df = pd.read_csv('./datasets/fashion-mnist_test.csv', sep = ',')

    train_data = np.array(train_df, dtype = 'float32')
    test_data = np.array(test_df, dtype='float32')

    x_train = train_data[:,1:]/255

    y_train = train_data[:,0]

    x_test = test_data[:,1:]/255

    y_test = test_data[:,0]

    x_train = x_train.reshape(x_train.shape[0], *image_shape)
    x_test = x_test.reshape(x_test.shape[0], *image_shape)

    return (x_train, y_train), (x_test, y_test)



def load_imdb():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    x_train = vectorize_sequences(train_data)

    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    return (x_train, y_train), (x_test, y_test)


def load_iris():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    y = to_categorical(y)

    return X, y


def load_mnist():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
                            
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return (train_images, train_labels), (test_images, (test_labels))

def load_sonar():
    data = pd.read_csv('sonar.csv')

    X = data.values[:,0:60].astype(float)
    y = [ 0 if x == "M" else 1 for x in data.values[:,60]]


    return X, y
