import numpy as np
from scipy import linalg


def load_chain():
    state_space = ('0', '1', '2', '3', '4')
    probability_space = np.array([
        [0, 0.3, 0.7, 0, 0],  # Tagus
        [0, 0, 1, 0, 0],  # Oeiras
        [0, 0, 0, 0.5, 0.5],  # Sete Rios
        [0, 0, 0, 0, 1],  # Praça de Londres
        [1, 0, 0, 0, 0],  # Alameda
    ])
    return (state_space, probability_space)


def prob_trajectory(M, sequence):
    state_space, transition_matrix = M[0], M[1]
    probability = 1
    for i in range(len(sequence) - 1):
        probability *= transition_matrix[state_space.index(sequence[i])][state_space.index(sequence[i + 1])]
    return probability


def stationary_dist(M):
    return linalg.eig(M[1], left=True, right=False)[1][:, 0]
