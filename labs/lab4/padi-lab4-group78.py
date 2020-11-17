import numpy as np
import matplotlib.pyplot as plt
import pdb

from sklearn import datasets as data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = data.load_iris()

data_x = iris.data[50:, :]  # Select only 2 classes
data_a = 2 * iris.target[50:] - 3  # Set output to {-1, +1}

# Get dimensions 
nE = data_x.shape[0]
nF = data_x.shape[1]


def preprocess(input_data, output_data):
    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.1, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler


def get_components(x_train, x_test, n_pc):
    pca = PCA(n_components=n_pc).fit(x_train)
    pca_train_x = pca.transform(x_train)
    pca_test_x = pca.transform(x_test)
    return pca_train_x, pca_test_x, pca


def lr(y, x, w):
    return 1 / (1 + np.exp(-y * w.T @ x))


def update_gradient(y_train, x_train_augmented, w):
    g = np.zeros((x_train_augmented.shape[1]))
    for n in range(x_train_augmented.shape[0]):
        y = y_train[n].reshape(1)
        X = x_train_augmented[n]
        g += y * X * (lr(y, X, w) - 1)
    return g / x_train_augmented.shape[1]


def update_hessian(y_train, x_train_augmented, w):
    H = np.zeros((x_train_augmented.shape[1], x_train_augmented.shape[1]))
    for n in range(x_train_augmented.shape[0]):
        y = y_train[n]
        X = x_train_augmented[n].reshape(x_train_augmented.shape[1], 1)
        p = lr(y, X, w)
        H += X @ X.T * p * (1 - p)
    return H


def my_train_lf(x_train, y_train):
    X_train_augmented = np.ones((x_train.shape[0], x_train.shape[1] + 1))  # +1 is the bias term
    X_train_augmented[:, :-1] = x_train[:, :]

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

    return w.reshape(X_train_augmented.shape[1], 1)


def skl_train_lr(train_x, train_y, C):
    model = LogisticRegression(solver='newton-cg', random_state=42, C=C)
    model.fit(train_x, train_y)
    return np.concatenate((model.coef_, model.intercept_.reshape(1, 1)), axis=1).reshape(train_x.shape[1] + 1, 1)


def nll(x, y, w):
    y = y.reshape(y.shape[0], 1)

    x_augmented = np.ones((x.shape[0], x.shape[1] + 1))  # +1 is the bias term
    x_augmented[:, :-1] = x[:, :]

    total = 0
    for n in range(x.shape[0]):
        total += np.log(lr(y[n], x_augmented[n], w))

    return total * -1 / x.shape[0]


x_train, y_train, x_test, y_test, scaler = preprocess(data_x, data_a)
pca_train_x, pca_test_x, pca = get_components(x_train, x_test, 2)
weights = my_train_lf(pca_train_x, y_train)

sk_weights = skl_train_lr(pca_train_x, y_train, 1e40)

nll(pca_train_x, y_train, weights)


def plot_overfit():
    COEFFS = [1e4, 9e3, 8e3, 7e3, 6e3, 5e3, 4e3, 3e3, 2e3,
              1000, 900, 800, 700, 600, 500, 400, 300, 200,
              100, 90, 80, 70, 60, 50, 40, 30, 20,
              10, 9, 8, 7, 6, 5, 4, 3, 2,
              1, .9, .8, .7, .6, .5, .4, .3, .2,
              .1, .09, .08, .07, .06, .05, .04, .03, .02, .01]

    RUNS = 1

    # Investigate overfitting
    err_train = np.zeros((len(COEFFS), 1))  # Error in training set
    err_valid = np.zeros((len(COEFFS), 1))  # Error in validation set

    np.random.seed(45)

    print('Completed: ', end='\r')

    for run in range(RUNS):

        # Split training data in "train" and "validation"
        x_train, x_valid, a_train, a_valid = train_test_split(pca_train_x, y_train, test_size=0.15)

        for n in range(len(COEFFS)):
            # Print progress
            print('Completed: %i %%' % int((run * len(COEFFS) + n) / (RUNS * len(COEFFS)) * 100), end='\r')

            # Train classifier
            w = skl_train_lr(x_train, a_train, COEFFS[n])

            # Compute train and test loss
            nll_train = nll(x_train, a_train, w)
            nll_valid = nll(x_valid, a_valid, w)

            err_train[n] += nll_train / RUNS
            err_valid[n] += nll_valid / RUNS

    print('Completed: 100%')

    plt.figure()
    plt.semilogx(COEFFS, err_train, 'k:', linewidth=3, label='Training')
    plt.semilogx(COEFFS, err_valid, 'k-', linewidth=3, label='Test')
    plt.xlabel('Regularization parameter')
    plt.ylabel('Negative log-likelihood')
    plt.legend(loc='best')

    plt.tight_layout()

    print(f'min nll: {np.min(err_valid, axis=0)[0]}')
    print(f'min coeff: {COEFFS[np.where(err_valid == np.min(err_valid, axis=0))[0][0]]}')

    plt.show()


plot_overfit()
