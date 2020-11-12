import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model

def run(fold):

    df = pd.read_csv('../input/adult_fold.csv')
    df = df.reset_index(drop=True)
    num_cols = ['fnlwgt', 'age', 'capital.gain', 'capital.loss', 'hours.per.week']
    df = df.drop(num_cols, axis=1)
    target_mapping = {'<=50K': 0, '>50K': 1}
    df.loc[:,'income'] = df.income.map(target_mapping)

    features = [f for f in df.columns if f not in ('income', 'kfold') ]
    
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
    model.fit(x_train, df_train.income.values)

    # we are using AUC so we want prob. values. We use prob of 1
    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(0,5):
        run(fold_)
