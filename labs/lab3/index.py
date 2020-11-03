import numpy as np

import pdb
import itertools

def load_pomdp(file_path, g):
    file = np.load(file_path)
    X, A, Z, P, O, c = file['X'], file['A'], file['Z'], file['P'], file['O'], file['c']
    return X, A, Z, tuple(P), tuple(O), c, g


M = load_pomdp('maze.npz', 0.95)

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


def sample_beliefs_one(POMDP, n):
    X, A, Z, P, O, c, g = POMDP

    x0 = np.random.choice(len(X))
    _, action_path, observation_path = gen_trajectory(POMDP, x0, n)

    beliefs = np.empty((n, 1, len(X)))
    belief = np.ones((1, len(X)), dtype=np.float64) / len(X)
    beliefs[0] = belief

    for i in range(1, n):
        estimated_belief = belief_update(P, O, belief, action_path[i], observation_path[i])
        beliefs[i, :, None] = estimated_belief[:]

    # print(f'n: {n}, beliefs shape {beliefs.shape}')

    _, ids = np.unique(beliefs, axis=0, return_index=True)
    ids.sort()
    beliefs = beliefs[ids]

    # print(f'n: {n}, beliefs shape {beliefs.shape}')

    # for combination in itertools.combinations(beliefs, 2):
    #     first, second = combination
    #     print(np.linalg.norm(first - second))
    #     print(np.linalg.norm(first - second) < 1e-3)
    #     if np.linalg.norm(first - second) < 1e-3:
    #         beliefs = np.delete(beliefs, np.where(np.all(beliefs == first, axis=2))[0], axis=0)

    idx_to_filter = []
    for combination in itertools.combinations(enumerate(beliefs), 2):
        first, second = combination
        first_index, first_element = first
        second_index, second_element = second

        # print(f'{first_index}, {second_index}, with norm = {np.linalg.norm(first_element - second_element)}, {np.linalg.norm(first_element - second_element, ord=1, keepdims=True, axis=1) < 1e-3}')

        # idx_to_filter.append(first_index)
 
        if np.linalg.norm(first_element - second_element, ord=1, keepdims=True, axis=1) < 1e-3:
            if first_index in idx_to_filter or second_index in idx_to_filter:
                pass
            elif first_index in idx_to_filter:
                idx_to_filter.append(second_index)
            else:
                idx_to_filter.append(first_index)
                # idx_to_filter.append(second_index)
    beliefs = np.delete(beliefs, idx_to_filter, axis=0)
    # print(idx_to_filter)

    # print(f'n: {n}, beliefs shape {beliefs.shape}')
    return beliefs


np.random.seed(42)

# 3 sample beliefs
B = sample_beliefs_one(M, 3)
print('%i beliefs sampled:' % len(B))
for i in range(len(B)):
    print(B[i])
    print('Belief adds to 1?', np.isclose(B[i].sum(), 1.))

B = sample_beliefs_one(M, 100)
print('%i beliefs sampled.' % len(B))
# for i in range(len(B)):
    # print(B[i])
    # print('Belief adds to 1?', np.isclose(B[i].sum(), 1.))

# print(t)
