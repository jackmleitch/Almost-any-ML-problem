import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from tensorflow.keras import layers, optimizers, callbacks, utils
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K

def create_model(data, catcols, numcols):
    """
    This function returns a compiled tf.keras model
    :param data: pandas dataframe
    :param catcols: list of categorical column names in dataframe
    :return: compiled tf.keras model 
    """

    # init list of inputs for embeddings
    inputs = []

    # init list of outputs for embeddings
    outputs = []

    # loop over all cat columns
    for c in catcols:
        # find number of unique values in column 
        num_unique_values = int(data[c].nunique())
        # we next calculate dimension of embedding
        # min size is half # of unique values
        # max size is 50
        embed_dim = int(min(np.ceil(num_unique_values/2),50))

        # simple keras input layer with size 1
        inp = layers.Input(shape=(1,))

        # add embedding layer to raw input 
        # embedding size is always 1 more than unique values in input
        out = layers.Embedding(
            num_unique_values + 1, embed_dim, name=c
            )(inp)

        # 1-d spacial dropout is the standard for embedding layers 
        out = layers.SpatialDropout1D(0.3)(out)

        # reshape input to dimension of embedding
        # this becomes our output layer for current feature
        out = layers.Reshape(target_shape=(embed_dim, ))(out)

        # add input to input list
        inputs.append(inp)

        # add output to output list
        outputs.append(out)


    # concatenate all output layers
    x = layers.Concatenate()(outputs)
    # x = layers.Flatten()(x)

    # num_data = layers.Input(shape=(data[numcols].shape[1],))
    # num_data = layers.BatchNormalization()(num_data)
    # num_data = layers.Flatten()(num_data)

    # x = layers.Concatenate()([x, num_data])   
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    y = layers.Dense(2, activation='softmax')(x)

    # create final model 
    model = Model(inputs=inputs, outputs=y)

    # compile the model 
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model 

def run(fold):

    df = pd.read_csv('../input/adult_fold.csv')

    target_mapping = {'<=50K': 0, '>50K': 1}
    df.loc[:,'income'] = df.income.map(target_mapping)

    num_cols = ['fnlwgt', 'age', 'capital.gain', 'capital.loss', 'hours.per.week']
    cat_cols = [c for c in df.columns if c  not in num_cols and c not in ('kfold', 'income')]
    
    features = [f for f in df.columns if f not in ('income', 'kfold') ]
    
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        df.loc[:, feat] = lbl_enc.fit_transform(df[feat].values)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    model = create_model(df, cat_cols, num_cols)


    # our features are lists of lists
    xtrain = [
        df_train[features].values[:, k] for k in range(len(features))
    ]
    xvalid = [
        df_valid[features].values[:, k] for k in range(len(features))
    ]

    ytrain = df_train.income.values
    yvalid = df_valid.income.values

    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    model.fit(xtrain, ytrain_cat, validation_data=(xvalid, yvalid_cat), verbose=1, batch_size=1024, epochs=3)

    valid_preds = model.predict(xvalid)[:, 1]

    print(metrics.roc_auc_score(yvalid, valid_preds))

    K.clear_session()

if __name__ == "__main__":
    for i in range(5):
        run(i)