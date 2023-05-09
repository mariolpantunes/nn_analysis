import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.datasets import cifar10, fashion_mnist
from textblob import TextBlob

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
    train_length = int(dataset.shape[0]*0.8)

    x_train = dataset[:train_length,:-1]
    y_train = dataset[:train_length,-1] -1

    x_test = dataset[train_length:,:-1]
    y_test = dataset[train_length:,-1] -1

    return (x_train, y_train), (x_test, y_test)

def load_higgs(): #Tabular classification numeric (high number of examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="clf_num/Higgs.csv", split="train")
    dataset = dataset.to_pandas().values
    train_length = int(dataset.shape[0]*0.8)

    x_train = dataset[:train_length,:-1]
    y_train = dataset[:train_length,-1]

    x_test = dataset[train_length:,:-1]
    y_test = dataset[train_length:,-1]

    return (x_train, y_train), (x_test, y_test)

def load_compas(): #Tabular classification categoric (low examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="clf_cat/compas-two-years.csv", split="train")
    dataset = dataset.to_pandas().values
    train_length = int(dataset.shape[0]*0.8)

    x_train = dataset[:train_length,:-1]
    y_train = dataset[:train_length,-1]

    x_test = dataset[train_length:,:-1]
    y_test = dataset[train_length:,-1]

    return (x_train, y_train), (x_test, y_test) 

def load_delays_zurich(): #Tabular regression numeric
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="reg_num/delays_zurich_transport.csv", split="train")
    dataset = dataset.to_pandas().values
    train_length = int(dataset.shape[0]*0.8)

    x_train = dataset[:train_length,:-1]
    y_train = dataset[:train_length,-1]

    x_test = dataset[train_length:,:-1]
    y_test = dataset[train_length:,-1]

    return (x_train, y_train), (x_test, y_test) 

def load_abalone(): #Tabular regression mixture (low examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="reg_cat/abalone.csv", split="train")
    dataset = dataset.to_pandas().values
    train_length = int(dataset.shape[0]*0.8)

    x_train = dataset[:train_length,:-1]
    y_train = dataset[:train_length,-1]

    x_test = dataset[train_length:,:-1]
    y_test = dataset[train_length:,-1]

    return (x_train, y_train), (x_test, y_test) 


def load_bike_sharing(): #Tabular regression mixture (numerous examples, more cat then reg)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="reg_cat/Bike_Sharing_Demand.csv", split="train")
    dataset = dataset.to_pandas().values
    train_length = int(dataset.shape[0]*0.8)

    x_train = dataset[:train_length,:-1]
    y_train = dataset[:train_length,-1]

    x_test = dataset[train_length:,:-1]
    y_test = dataset[train_length:,-1]

    return (x_train, y_train), (x_test, y_test) 


def load_sms_spam_collection(): 

    dataset = pd.read_csv("..\\nn_analysis\\datasets\\SMSSpamCollection", sep='\t', header=None, names=["v1", "v2"])

    #lower case
    dataset["v2"] = dataset["v2"].str.lower()
    #tokenization
    dataset["v2"] = dataset.apply(lambda x: nltk.word_tokenize(x["v2"]), axis=1)
    #stemming 
    stemmer = PorterStemmer()
    dataset["v2"] = dataset["v2"].apply(lambda x: [stemmer.stem(y) for y in x])
    #remove stop words
    stop_words = stopwords.words("english")
    dataset["v2"].apply(lambda x: [y for y in x if y not in stop_words])
    #get polarity and subjectivity 
    def get_sentiment(words):
        blob = TextBlob(','.join(words))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        return polarity, subjectivity

    dataset[["polarity", "subjectivity"]] = dataset["v2"].apply(lambda x: pd.Series(get_sentiment(x)))

    #Convert label str -> bool 
    dataset["v1"] = dataset["v1"].replace("spam", 1)
    dataset["v1"] = dataset["v1"].replace("ham", 0)

    #TF-IDF
    dataset["v2"] = dataset["v2"].apply(lambda x: ' '.join(x))

    tfidf = TfidfVectorizer()

    tfidf.fit(dataset["v2"])

    tfidf_data = tfidf.transform(dataset["v2"])

    #Final dataframe 
    tfidf_df = pd.DataFrame(tfidf_data.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_df.insert(0, "v1", dataset["v1"])
    tfidf_df.insert(1, "polarity", dataset["polarity"])
    tfidf_df.insert(2, "subjectivity", dataset["subjectivity"])

    train_length = int(tfidf_df.shape[0]*0.8)

    x_train = tfidf_df.iloc[:train_length,1:tfidf_df.shape[1]].values
    y_train = tfidf_df.iloc[:train_length, 0].values

    x_test = tfidf_df.iloc[train_length: , 1:tfidf_df.shape[1]].values
    y_test = tfidf_df.iloc[train_length: , 0].values

    return (x_train, y_train), (x_test, y_test) 