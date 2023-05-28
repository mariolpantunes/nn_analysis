import numpy as np
from sklearn.metrics import make_scorer, matthews_corrcoef, mean_squared_error


def mcc():

    def mcc(y_true, y_pred):
        if type(y_true[0]) != np.float64:
            y_true =[np.argmax(x) for x in y_true]
        if type(y_pred[0]) != np.float64:
            y_pred =[np.argmax(x) for x in y_pred]

        return matthews_corrcoef(y_true, y_pred)
    
    return make_scorer(mcc, greater_is_better=True)



def mse():

    def mse(y_true, y_pred):

        return mean_squared_error(y_true, y_pred)
    
    return make_scorer(mse, greater_is_better=False)