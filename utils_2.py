import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

def read_normalize_data(filename):

    data = pd.read_csv(filename).as_matrix()

    X = data[:, 1:]
    X_std = X.std(axis = 0)
    np.place(X_std, X_std == 0, 1)
    X = (X - X.mean(axis = 0)) / X_std
    Y = data[:, 0]

    return X, Y

def produce_PCA(filename, num_parameters, show_fig = True):

    X, Y = read_normalize_data(filename)

    pca = PCA(n_components=300)

    pca.fit(X)
    variables = pca.explained_variance_ratio_

    cum_sum = np.cumsum(variables)

    if show_fig:
        _ = plt.plot(cum_sum)
        plt.show()

    X_PCA_transformed = pca.transform(X)
    print(X_PCA_transformed.shape)

    return X_PCA_transformed, Y


def onehotencode(Y):
    N = len(Y)
    K = len(set(Y))
    onehot = np.zeros((N, K))
    onehot[np.arange(N), Y] = 1
    return onehot

#Get the data.
def get_data(PCA_data, num_parameters):

    if PCA_data == True:
        X, Y = produce_PCA("train.csv", num_parameters, show_fig = False)
        X, Y = shuffle(X, Y)
        #We will also return the size of the X dataset
        N, D = X.shape
    else:
        X, Y = read_normalize_data("train.csv")
        #We will also return the size of the X dataset
        X, Y = shuffle(X, Y)
        N, D = X.shape

    return X, Y, N, D
