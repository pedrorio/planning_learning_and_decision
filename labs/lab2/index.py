import numpy as np
import numpy.random as rand
import time

np.set_printoptions(precision=3)

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


def evaluate_pol(MDP, PI):
    X, A, P, c, g = MDP
    X = np.expand_dims(X, axis=1)
    A = np.expand_dims(A, axis=1)

    P_PI = np.zeros((len(X), len(X)))
    c_PI = np.zeros((len(X), 1))

    for i in range(len(A)):
        P_PI += np.diag(PI[:, i]).dot(P[i])
        c_PI += np.diag(PI[:, i]).dot(c[:, i, None])

    J = np.matmul(np.linalg.inv(np.eye((len(X))) - g * P_PI), c_PI)
    return J


def value_iteration(MDP):
    X, A, P, c, g = MDP
    J = np.zeros(len(X))
    err = 1
    i = 0

    start = time.time()
    while err > pow(10, -8):
        Qu = c[:, 0] + g * P[0].dot(J)
        Qd = c[:, 1] + g * P[1].dot(J)
        Ql = c[:, 2] + g * P[2].dot(J)
        Qr = c[:, 3] + g * P[3].dot(J)
        J_new = np.min((Qu, Qd, Ql, Qr), axis=0)
        err = np.linalg.norm(J_new - J)
        i += 1
        J = J_new
    end = time.time()
    print(f'Execution time: {round((end - start), 3)}')
    print(f'N. iterations: {i}')
    return J


# Add your code here.
import time


def policy_iteration(MDP):
    X, A, P, c, g = MDP

    # Initial policy
    PI = np.ones((len(X), len(A))) / len(A)
    quit = False
    i = 0

    C_PI = np.zeros((len(X), 1))
    P_PI = np.zeros((len(X), len(X)))
    Q = np.zeros((len(X), len(A)))

    start = time.time()

    while not quit:
        for j in range(len(A)):
            C_PI += np.diag(PI[:, j]).dot(c[:, j, None])
            P_PI += np.diag(PI[:, j]).dot(P[j])

        J = np.linalg.inv(np.eye(len(X)) - g * P_PI).dot(C_PI)

        for i in range(len(A)):
            Q[:, i, None] = c[:, i, None] + g * P[i].dot(J)

        PI_new = np.zeros((len(X), len(A)))

        for i in range(len(A)):
            PI_new[:, i, None] = np.isclose(
                Q[:, i, None],
                np.min(Q, axis=1, keepdims=True),
                atol=pow(10, -8),
                rtol=pow(10, -8)
            ).astype(int)

        PI_new = PI_new / np.sum(PI_new, axis=1, keepdims=True)
        quit = (PI == PI_new).all()
        PI = PI_new
        i += 1

    end = time.time()
    print(f'Execution time: {round((end - start), 3)}')
    print(f'N. iterations: {i}')
    return PI

def simulate(MDP, PI, x0, length):
    X, A, P, c, g = MDP
    # path = []
    # initial state
    # X[x0]
    # action probabilities
    # PI[x0]
    # action is a random choice given the probabilities
    # a = random.choice(A, PI[x0])
    # Transition Probabilities
    # P[a]
    # x is a random choice given the probabilities
    # x = random_choice(X, P[a])
    # path.append([a, x])
    # complete the path

    # discount the cost
    # start at the end
    # get the cost c[x][a]
    # discount it and add the previous c[x][a]
    # reach the beginning

    # repeat for n paths
    # compute the average

