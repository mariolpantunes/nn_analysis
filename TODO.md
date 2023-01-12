
# In models
- Define a correct prebuilt model for classification/regression
    - Define loss function
    - Define optimizer
    - Define activation for last layer
    - Check if there are other params that can be static
    
    - update model construction
        - CNN/RNN/Dense can have a dropout
        - Number of dense layers after CNNs is a hyperparameter - DONE
        - number nodes in each layer is now an hyperparameter (both CNN, dense, RNN ) - DONE
        - GRU changes to RNN and layers can be LSTM or GRU (verify relevant hyperparameters) - DONE
        - pass the model, classification, output and dataset in the hyperparameter file. - Done

# For search (hyperparameters folder)
- Define a predifine set of hyperparameters for each dataset (similar to cifar10)
- predefine a number of random seeds to apply
- verify callbacks to save momentaneous research

# In datasets
- Create different preprocessings for each type of model and each dataset.
- Define the pool of datasets used.


