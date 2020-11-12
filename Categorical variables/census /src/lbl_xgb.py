import itertools
import pandas as pd
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

def feature_engineering(df, cat_cols):
    '''Takes in dataframe and categorical column titles and creates interaction columns'''
    combi = list(itertools.combinations(cat_cols,2))
    for c1, c2 in combi:
        df.loc[:, c1 + '_' + c2] = df[c1].astype(str) + '_' + df[c2].astype(str)
    return df

def run(fold):
    df = pd.read_csv('../input/adult_fold.csv')
    df = df.reset_index(drop=True)
    num_cols = ['fnlwgt', 'age', 'capital.gain', 'capital.loss', 'hours.per.week']
    # df = df.drop(num_cols, axis=1)
    target_mapping = {'<=50K': 0, '>50K': 1}
    df.loc[:,'income'] = df.income.map(target_mapping)

    cat_cols = [c for c in df.columns if c  not in num_cols and c not in ('kfold', 'income')]
    df = feature_engineering(df, cat_cols)

    features = [f for f in df.columns if f not in ('income', 'kfold') ]
    
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:,col] = lbl.transform(df[col])

    # get training data using folds
    df_train =df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(
        n_jobs=-1
    )
    model.fit(x_train, df_train.income.values)

    # we are using AUC so we want prob. values. We use prob of 1
    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(0,5):
        run(fold_)
