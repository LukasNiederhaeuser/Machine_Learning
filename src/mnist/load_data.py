import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml


def load_mnist_data():

    # variable for shuffling of index
    shuffle_index = np.random.permutation(60000)

    # Load mnist data
    mnist = fetch_openml(name='mnist_784', version=1)
    # Create features and target
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    # Create train and test split
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    # Shuffle the index
    X_train, y_train = X_train.iloc[shuffle_index], y_train.iloc[shuffle_index]

    return X_train, X_test, y_train, y_test