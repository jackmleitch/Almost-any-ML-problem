import pandas as pd

df = pd.read_csv('../input/train_fold.csv')
print(df['kfold'].value_counts())

for i in range(0, 5):
    print(df[df.kfold == i].target.value_counts())