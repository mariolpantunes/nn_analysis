
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy

opto = "adam" #["adam","adadelta","adagrad","adamax","nadam","ftrl","sgd","rmsprop"]

def create_dense_model():
    model = Sequential()
    model.add(Dense(units=10, input_dim=10,activation='relu'))
    model.add(Dense(units=1,activation='linear'))
    model.compile(optimizer=opto, loss='mean_squared_error')

    return model


def create_cnn_model():

    loss_function = sparse_categorical_crossentropy
    no_classes = 10

    input_shape = (32, 32, 3)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))

    model.compile(loss=loss_function, optimizer=opto)

    return model

def create_fashion_model():
    image_shape = (28,28,1) 

    cnn_model = Sequential([
        Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape),
        MaxPooling2D(pool_size=2) ,
        Dropout(0.2),
        Flatten(),
        Dense(32,activation='relu'),
        Dense(10,activation = 'softmax')
        
    ])

    cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=opto)

def create_imdb_model():

    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(10000,)))        
    model.add(Dense(16, activation='relu'))                     
    model.add(Dense(1, activation='sigmoid'))
                                                                                

    model.compile(optimizer=opto,                          
                loss='binary_crossentropy')

def create_IRIS_model():

    network = Sequential()
    network.add(Dense(512, activation='relu', input_shape=(4,)))
    network.add(Dense(3, activation='softmax'))

    network.compile(optimizer=opto,
                    loss='categorical_crossentropy')

def create_mnist_model():

    model = Sequential() 
    model.add(Dense(512, activation= 'relu', input_shape=(28 * 28,)))
    model.add(Dense(10, activation='softmax' ))    

    model.compile(optimizer= opto,                                  
            loss='categorical_crossentropy')

def create_sonar_model():
    model = Sequential()
    model.add(Dense(60, input_dim=60, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=opto)