import numpy as np
from sklearn.metrics import make_scorer, matthews_corrcoef


def mcc():

    def mcc(y_pred, y_true):
        y_true =[np.argmax(x) for x in y_true]
        y_pred =[np.argmax(x) for x in y_pred]

        return matthews_corrcoef(y_true, y_pred)
    
    return make_scorer(mcc, greater_is_better=True)