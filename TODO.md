
# In models
- Define a correct prebuilt model for classification/regression
    - Define loss function
    - Define optimizer
    - Define activation for last layer
    - Check if there are other params that can be static

# For search (hyperparameters folder)
- Define a predifine set of hyperparameters for each dataset (similar to cifar10)

# In run
- Define the preprocessing specific for each model and task (how we deal with multidimensional data etc)
    - Similar to what we have in CIFAR10 (even though its just a dummy example)

- Define the correct set of metrics for classification and regression
    - Right now we have MCC RMSE

# In datasets
-  Define a dataset load from a csv file.
    - Make a check to see if the .csv is in the correct format. (not sure how)

