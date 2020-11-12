import copy 
import pandas as pd

from sklearn import metrics 
from sklearn import preprocessing
import xgboost as xgb

def mean_target_encoding(data):
    
    # make copy of dataframe
    df = copy.deepcopy(data)

    # list of numerical columns 
    num_cols = ['fnlwgt', 'age', 'capital.gain', 'capital.loss', 'hours.per.week']

    # map targets to 0s and 1s
    target_mapping = {'<=50K': 0, '>50K': 1}
    df.loc[:, 'income'] = df.income.map(target_mapping)

    # get features we want to encode 
    features = [f for f in df.columns if f not in ('income', 'kfold') and f not in num_cols]

    # fill in NAs with NONE 
    for col in features:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna('NONE')
    
    # we label encode all the features
    for col in features:
        if col not in num_cols:
            # initialise label encoder
            lbl = preprocessing.LabelEncoder()
            # fit label encoder
            lbl.fit(df[col])
            # transform all of the data
            df.loc[:, col] = lbl.transform(df[col])

    # a list to store 5 validation dataframes 
    encoded_dfs = []

    # loop over every fold 
    for fold in range(5):
        # get training and validation data
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        # for all cat columns
        for column in features:
            # create dict of category:mean target 
            mapping_dict = dict(
                df_train.groupby(column)['income'].mean()
            )
            # column_enc is the new column we have with mean encodings
            df_valid.loc[:, column + '_enc'] = df_valid[column].map(mapping_dict)
        # append to our list of encoded validation dfs
        encoded_dfs.append(df_valid)
    # create full dataframe again and return 
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df
    

def run(df, fold):
    
    df_train =df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    features = [f for f in df.columns if f not in ('kfold', 'income')]

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7
    )
    model.fit(x_train, df_train.income.values)

    # we are using AUC so we want prob. values. We use prob of 1
    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    # read data
    df = pd.read_csv('../input/adult_fold.csv')
    # create mean target encoded cats and munge data 
    df = mean_target_encoding(df)
    # run training and validation for 5 folds
    for fold_ in range(5):
        run(df, fold_)

