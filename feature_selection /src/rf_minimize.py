import numpy as np
import pandas as pd

from functools import partial 

from sklearn import ensemble, metrics, model_selection

from skopt import gp_minimize, space

def optimize(params, param_names, x, y): 
    """
    The main optimize function. 
    This function takes all the arguments from the search space
    and training features and targets. It then initializes the models
    by setting the chosen params and runs cross-val and returns negative
    accuracy score. 
    :param params: list of params from gp-minimize 
    :param param_names: list of (ordered) param names
    :param x: training data
    :param y: labels/targets
    :return: negative accuracy after 5 folds
    """
    # convert params to dictionary 
    params = dict(zip(param_names, params))

    # initialize model with current params
    model = ensemble.RandomForestClassifier(**params)

    # initialize stratified k-fold
    kf = model_selection.StratifiedKFold(n_splits=5)

    # initialize accuracy list
    accuracies = []

    # loop over all folds
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        # fit model for current fold
        model.fit(xtrain, ytrain)

        # create preds
        preds = model.predict(xtest)

        # calc and append accuracy 
        fold_accuracy = metrics.accuracy_score(
            ytest,
            preds
        )
        accuracies.append(fold_accuracy)
    
    # return negative accuracy 
    return -1 * np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv('../input/mobile_train.csv')

    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    # define a parameter space
    param_space = [
        space.Integer(3, 15, name='max_depth'),
        space.Integer(100, 1500, name="n_estimators"),
        space.Categorical(['gini', 'entropy'], name='criterion'),
        space.Real(0.01, 1, prior='uniform', name='max_features')
    ]

    # make a list of param names
    param_names = [
        'max_depth',
        'n_estimators',
        'criterion',
        'max_features'
    ]

    # functools partial creates a new function which has same params as the 
    # optimize function except for the fact that only one param, 
    # i.e. the 'params' parameter is required.
    # this is how gp_minimise expects the optimization function to be.
    optimization_function = partial(
        optimize,
        param_names=param_names,
        x=X,
        y=y 
        )
    
    # we now call gp_minimize func from scikit-optimize
    # it used bayesian optimization to minimimize the optimize func.
    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=20,
        n_random_starts=10,
        verbose=10
    )

    # create best params dict
    best_params = dict(
        zip(param_names, result.x)
    )
    print(best_params)

from skopt.plots import plot_convergence
plot_convergence(result)
