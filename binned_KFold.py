import pandas as pd 
import numpy as np
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv('train.csv', sep=';')

    # create new column 'kfold' with val -1 
    df['kfold'] = -1
    df = df.sample(frac = 1).reset_index(drop=True)

    # number of bins according to Sturge's law
    num_bins = int(np.floor(1+np.log2(len(data))))

    # bin target 
    df.loc[:, 'bins'] = pd.cut(data['target'], bins = num_bins, labels = False)

    # initialise kfold class
    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=df.bins.values)):
        df.loc[v_, 'kfold'] = f
    
    df = df.drop('bins', axis=1)
    df.to_csv('train_fold.csv', index=False)
