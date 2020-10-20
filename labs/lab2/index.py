import numpy as np

np.random.seed(42)


def load_mdp(filename, gamma):
    file = np.load(filename)
    X, A, P, c = file['X'], file['A'], file['P'], file['c']
    return X, A, tuple(P), c, gamma


def noisy_policy(MDP, a, eps):
    _, A, *_ = MDP
    n = len(A)
    policy = np.ones(n)
    policy *= eps / (n - 1)
    policy[a] = 1 - eps
    return policy
