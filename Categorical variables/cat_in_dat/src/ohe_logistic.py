import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
import argparse

def run(fold):
    df = pd.read_csv('../input/train_fold.csv')
    features = [f for f in df.columns if f not in ('id', 'target', 'kfold') ]
    
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # get training data using folds
    df_train =df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data[features])

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    model = linear_model.LogisticRegression()
    model.fit(x_train, df_train.target.values)

    # we are using AUC so we want prob. values. We use prob of 1
    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    # # initialize ArgumentParser class of argparse
    # parser = argparse.ArgumentParser()

    # # add different arguments and their types
    # parser.add_argument('--fold', type=int)
    # # read arguments from command line
    # args = parser.parse_args()

    # run the fold specified by command line arguments
    for fold_ in range(0,5):
        run(fold_)
