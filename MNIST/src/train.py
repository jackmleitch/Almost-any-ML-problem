# /Users/Jack/Documents/ML/Projects/Approach_any_ML-/MNIST/src
import os
import argparse
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

import config
import model_dispatcher

def run(fold, model):
    # read in data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # train data is when fold isnt equal to provided fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop target column (label) and turn to np array 
    x_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values

    # same for validation 
    x_valid = df_valid.drop('label', axis=1).values
    y_valid = df_valid.label.values

    # fetch model from model dispatched
    clf = model_dispatcher.models[model]

    # fit model on training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # calculate accuracy (label distribution is uniform)
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    # save the model
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f"../models/dt_{fold}.bin")
    )

if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add different arguments and their types
    parser.add_argument('--fold', type=int)
    parser.add_argument('--model', type=str)
    
    # read arguments from command line
    args = parser.parse_args()

    # run the fold specified by command line arguments
    run(
        fold=args.fold,
        model=args.model
        )

