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
    Q = np.zeros((len(X), len(A)))
    err = 1
    i = 0

    start = time.time()
    while err > pow(10, -8):
        for i in range(len(A)):
            Q[:, i, None] = c[:, i, None] + g * P[i].dot(J)

        J_new = np.min(Q, axis=1, keepdims=True)
        err = np.linalg.norm(J_new - J)
        i += 1
        J = J_new
    end = time.time()
    print(f'Execution time: {round((end - start), 3)}')
    print(f'N. iterations: {i}')
    return J


def policy_iteration(MDP):
    X, A, P, c, g = MDP

    X = np.expand_dims(X, axis=1)
    A = np.expand_dims(A, axis=1)

    PI = np.ones((len(X), len(A))) / len(A)
    quit = False
    i = 0

    # C_PI = np.zeros((len(X), 1))
    # P_PI = np.zeros((len(X), len(X)))
    Q = np.zeros((len(X), len(A)))

    def J_Optimum(MDP):
        X, A, P, c, g = MDP
        J = np.zeros((len(X), 1))
        Q = np.zeros((len(X), len(A)))
        err = 1
        i = 0

        while err > 1e-8:
            for i in range(len(A)):
                Q[:, i, None] = c[:, i, None] + g * P[i].dot(J)

            Jnew = np.min(Q, axis=1, keepdims=True)
            err = np.linalg.norm(Jnew - J)
            i += 1
            J = Jnew
        return J

    start = time.time()

    while not quit:
        # for j in range(len(A)):
        #     C_PI += np.diag(PI[:, j]).dot(c[:, j, None])
        #     P_PI += np.diag(PI[:, j]).dot(P[j])
        # J = np.linalg.inv(np.eye(len(X)) - g * P_PI).dot(C_PI)
        J = J_Optimum(MDP)
        print(J.shape)

        for i in range(len(A)):
            Q[:, i, None] = c[:, i, None] + g * P[i].dot(J)

        PI_new = np.zeros((len(X), len(A)))

        for i in range(len(A)):
            PI_new[:, i, None] = np.isclose(
                Q[:, i, None],
                np.min(Q, axis=1, keepdims=True),
                atol=1e-8,
                rtol=1e-8
            ).astype(int)

        PI_new = PI_new / np.sum(PI_new, axis=1, keepdims=True)
        quit = (PI == PI_new).all()
        PI = PI_new
        i += 1

    end = time.time()
    print(f'Execution time: {round((end - start), 3)}')
    print(f'N. iterations: {i}')
    return PI

NRUNS =100

def simulate(MDP, PI, x0, length):
    X, A, P, c, g = MDP

    discounted_costs = []
    for run in range(NRUNS):
        path = []

        x = x0
        a = np.random.choice(len(A), p=PI[x])
        path.append((x, a))

        for i in range(1,length):
            x = np.random.choice(len(X), p=P[a][x])
            a = np.random.choice(len(A), p=PI[x])
            path.append((x, a))

        discounted_cost = 0
        for j in range(length):
            x, a = path[j]
            discounted_cost += c[x][a] * pow(g,j)

        discounted_costs.append(discounted_cost)
    return sum(discounted_costs)/len(discounted_costs)