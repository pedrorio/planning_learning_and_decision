import numpy as np
import matplotlib.pyplot as plt

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
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_A, test_size=0.1, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler


def get_components(X_train, X_test, n_pc):
    pca = PCA(n_components=n_pc).fit(X_train)
    pca_train_x = pca.transform(X_train)
    pca_test_x = pca.transform(X_test)
    return pca_train_x, pca_test_x, pca

def my_train_lf(X_train, y_train):
	
