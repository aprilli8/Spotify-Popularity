import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import pickle

def make_simulated_data():
    """
    Make and return simulated data
    """

    Xmat, Y = make_classification(n_samples=75, n_features=10, n_redundant=5, random_state=0)
    Xmat_train, Xmat_val, Y_train, Y_val = train_test_split(Xmat, Y, test_size=0.33, random_state=42)
    means = np.mean(Xmat_train, axis=0)
    stdevs = np.std(Xmat_train, axis=0)
    Xmat_train = (Xmat_train - means)/stdevs
    Xmat_val = (Xmat_val - means)/stdevs

    return Xmat_train, Xmat_val, Y_train, Y_val

