import numpy as np

from functools import partial
from scripy.optimize import fmin
from sklearn import metrics


class OptimizeStacking:
    """
    Class for optimizing AUC
    This class finds the best weights for 
    any model/any metric/ any type of prediction
    """

    def __init__(self):
        self.coef_ = 0

    def _auc(self, coef, X, y):
        """
        Calculates and returns AUC.
        :param coef: coef list, same length as the number of models
        :param X: preds (2D array)
        :param y: targets (1D array)
        :return: negative AUC score
        """
        # element wise multiplication of preds and coefficients
        x_coef = X * coef
        # preds by row wise sum
        predictions = np.sum(x_coef, axis=1)
        # calc. AUC score
        auc_score = metrics.roc_auc_score(y, predictions)
        # return neg AUC
        return -1.0 * auc_score

    def fit(self, X, y):
        loss_partial = partial(self._auc, X=X, y=y)
        # use dirichlet distribution to init. the coefficients
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)
        # minimise the loss function
        self.coef_ = fmin(loss_partial, initial_coef, disp=True)

    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions

