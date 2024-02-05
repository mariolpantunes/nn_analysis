import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.datasets import cifar10, fashion_mnist
from textblob import TextBlob
from sklearn.model_selection import train_test_split

from datasets import load_dataset

'''
    Loads a csv file in the format feature1, feature2, ..., label.
    Divides it into 80% for training and 20% for testing.
    Returns ((x_train, y_train), (x_test, y_test))
'''
def load_dataset_from_file(file_location): 

    if file_location.split(".")[-1] != "csv":
        raise ValueError("Dataset should be a csv file.")

    data = pd.read_csv(file_location, index_col=False)

    train_length = int(data.shape[0]*0.8)

    x_train = data.iloc[:train_length,:data.shape[1]-1].values
    y_train = data.iloc[:train_length, data.shape[1]-1].values

    x_test = data.iloc[train_length: , :data.shape[1]-1].values
    y_test = data.iloc[train_length: , data.shape[1]-1].values

    return (x_train, y_train), (x_test, y_test)

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

def load_covertype(): #Tabular classification categoric_numeric (high number of examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="clf_cat/covertype.csv", split="train")
    dataset = dataset.to_pandas().values

    x = dataset[:,:-1]
    y = dataset[:,-1] -1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_test, y_test)

def load_higgs(): #Tabular classification numeric (high number of examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="clf_num/Higgs.csv", split="train")
    dataset = dataset.to_pandas().values

    x = dataset[:,:-1]
    y = dataset[:,-1] 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_test, y_test)

def load_compas(): #Tabular classification categoric (low examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="clf_cat/compas-two-years.csv", split="train")
    dataset = dataset.to_pandas().values
    
    x = dataset[:,:-1]
    y = dataset[:,-1] 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_test, y_test)

def load_delays_zurich(): #Tabular regression numeric
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="reg_num/delays_zurich_transport.csv", split="train")
    dataset = dataset.to_pandas().values
    
    x = dataset[:,:-1]
    y = dataset[:,-1] 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_test, y_test)

def load_abalone(): #Tabular regression mixture (low examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="reg_cat/abalone.csv", split="train")
    dataset = dataset.to_pandas().values
    
    x = dataset[:,:-1]
    y = dataset[:,-1] 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_test, y_test)


def load_bike_sharing(): #Tabular regression mixture (numerous examples, more cat then reg)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="reg_cat/Bike_Sharing_Demand.csv", split="train")
    dataset = dataset.to_pandas().values
    
    x = dataset[:,:-1]
    y = dataset[:,-1] 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_test, y_test)