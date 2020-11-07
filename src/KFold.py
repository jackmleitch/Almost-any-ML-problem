import pandas as pd 
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv('/Users/Jack/Documents/ML/Projects/Approach_any_ML-/input/winequality-red.csv', sep=';')

    # create new column 'kfold' with val -1 
    df['kfold'] = -1
    df = df.sample(frac = 1).reset_index(drop=True)

    # initialise kfold class
    kf = model_selection.KFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
    
    df.to_csv('/Users/Jack/Documents/ML/Projects/Approach_any_ML-/input/train_fold.csv', index=False)
