
# In models
- Define a correct prebuilt model for classification/regression
    - Define loss function
    - Define optimizer
    - Define activation for last layer
    - Check if there are other params that can be static

# In search
- Define a predifine set of hyperparameters for each dataset (similar to cifar10)

# In run
- Define the preprocessing specific for each model
    - Similar to what we have in CIFAR10 (even though its just a dummy example)

- Define the correct set of metrics for classification and regression
    - Right now we only have MCC defined

- Define user input validation
    - Check that folders exist and are not empty for hyperparameter folders
    - Check that folder exists for results
    - Check if the selected model exists
    - Check if classification is True or False
    - Check if file exists when dataset is not supported. 

# In datasets
-  Define a dataset load from a csv file.
    - Define a common structure to load datasets which include for example the abalone dataset.
    - change the abalone to match such standard .csv format
    - Make a check to see if the .csv is in the correct format.

