import numpy as np
# import matplotlib.pyplot as plt
import pdb

from sklearn import datasets as data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
iris = data.load_iris()

data_X = iris.data[50:, :]  # Select only 2 classes
data_A = 2 * iris.target[50:] - 3  # Set output to {-1, +1}

# Get dimensions 
nE = data_X.shape[0]
nF = data_X.shape[1]

def preprocess(input_data, output_data):
    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.1, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler


def get_components(X_train, X_test, n_pc):
    pca = PCA(n_components=n_pc).fit(X_train)
    pca_train_x = pca.transform(X_train)
    pca_test_x = pca.transform(X_test)
    return pca_train_x, pca_test_x, pca


def compute_probability(y, X, w):
    return 1 / (1 + np.exp(-y * w.T @ X))


def update_gradient(y_train, X_train_augmented, w):
    g = np.zeros((X_train_augmented.shape[1]))
    for n in range(X_train_augmented.shape[0]):
        y = y_train[n].reshape(1)
        X = X_train_augmented[n]
        g += y * X * (compute_probability(y, X, w) - 1)
    return g / X_train_augmented.shape[1]


def update_hessian(y_train, X_train_augmented, w):
    H = np.zeros((X_train_augmented.shape[1], X_train_augmented.shape[1]))
    for n in range(X_train_augmented.shape[0]):
        y = y_train[n]
        X = X_train_augmented[n].reshape(X_train_augmented.shape[1], 1)
        p = compute_probability(y, X, w)
        H += X @ X.T * p * (1 - p)
    return H


def my_train_lf(X_train, y_train):
    X_train_augmented = np.ones((X_train.shape[0], X_train.shape[1] + 1))  # +1 is the bias term
    X_train_augmented[:, :-1] = X_train[:, :]

    w = np.zeros((X_train_augmented.shape[1]))

    g = update_gradient(y_train, X_train_augmented, w)
    H = update_hessian(y_train, X_train_augmented, w)
    delta = np.linalg.norm(w - np.linalg.inv(H) @ g - w)
    epsilon = 10e-5

    while delta > epsilon:
        g = update_gradient(y_train, X_train_augmented, w)
        H = update_hessian(y_train, X_train_augmented, w)
        new_w = w - np.linalg.inv(H) @ g
        delta = np.linalg.norm(new_w - w)
        w = new_w

    return w


X_train, y_train, X_test, y_test, scaler = preprocess(data_X, data_A)
pca_train_x, pca_test_x, pca = get_components(X_train, X_test, 2)
weights = my_train_lf(pca_train_x, y_train)
