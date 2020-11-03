import numpy as np

def load_pomdp(file_path, g):
    file = np.load(file_path)
    X, A, Z, P, O, c = file['X'], file['A'], file['Z'], file['P'], file['O'], file['c']
    return X, A, Z, tuple(P), tuple(O), c, g

def gen_trajectory(POMDP, x0, n):
    X, A, Z, P, O, c, g = POMDP

    state_path = np.zeros(n + 1, dtype=int)
    action_path = np.zeros(n, dtype=int)
    observation_path = np.zeros(n, dtype=int)

    x = x0
    state_path[0] = x

    for i in range(0, n):
        a = np.random.choice(len(A))
        action_path[i] = a

        x = np.random.choice(len(X), p=P[a][x, :])
        state_path[i + 1] = x

        z = np.random.choice(len(Z), p=O[a][x, :])
        observation_path[i] = z
  
    return state_path, action_path, observation_path

def belief_update(P, O, b, a, z):
    estimated_belief = b @ P[a] @ np.diag(O[a][:, z])
    return estimated_belief / np.linalg.norm(estimated_belief, ord=1, keepdims=True, axis=1)

def sample_beliefs(POMDP, n):
    X, A, Z, P, O, c, g = POMDP

    x0 = np.random.choice(len(X))
    _, action_path, observation_path = gen_trajectory(POMDP, x0, n)

    beliefs = np.empty((n, 1, len(X)))
    beliefs[0] = np.ones((1, len(X)), dtype=np.float64) / len(X)

    ids_to_delete = []

    for i in range(1, n):
        estimated_belief = belief_update(P, O, beliefs[i-1], action_path[i], observation_path[i])
        for j in range(0, i):
            if np.linalg.norm(estimated_belief - beliefs[j], ord=1, keepdims=True, axis=1) >= 1e-3:
                beliefs[i, :, None] = estimated_belief[:]
            else:
                ids_to_delete.append(i)

    beliefs = np.delete(beliefs, ids_to_delete, axis=0)
    return beliefs

def solve_mdp(POMDP):
    X, A, Z, P, O, c, g = POMDP

    J = np.zeros(len(X))
    Q = np.empty((len(X), len(A)))

    while True:
        for a in range(len(A)):
            Q[:, a] = c[:, a] + g * (P[a] * J).sum(axis=1)
        J_new = np.min(Q, axis=1)
        if np.linalg.norm(J_new - J) < 1e-8:
            break
        else:
            J = J_new
    return Q

def get_heuristic_action(belief, Q, heuristic):
    Qmin = Q.min(axis=1, keepdims=True)
    epsilon = 1e-8
    policy_mdp = np.isclose(Q, Qmin, atol=epsilon, rtol=epsilon).astype(int)
    policy_mdp = policy_mdp / policy_mdp.sum(axis=1, keepdims=True)

    if heuristic == "mls":
        policy = np.argmax(policy_mdp[np.argmax(belief)])
    elif heuristic == "av":
        policy = np.argmax(belief @ policy_mdp, axis=1)[0]
    elif heuristic == "q-mdp":
        policy = np.argmin(belief @ Q)

    return policy

def get_optimal_action(belief, G, ids):
    policy = belief @ G
    return ids[np.argmin(policy)]