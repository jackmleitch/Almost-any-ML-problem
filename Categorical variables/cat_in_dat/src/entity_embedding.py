import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from tensorflow.keras import layers, optimizers, callbacks, utils
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K

def create_model(data, catcols):
    """
    This function returns a compiled tf.keras model
    """
