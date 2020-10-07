import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


def load_chain():
    state_space = ('0', '1', '2', '3', '4')
    probability_space = np.array([
        [0, 0.3, 0.7, 0, 0],  # Tagus
        [0, 0, 1, 0, 0],  # Oeiras
        [0, 0, 0, 0.5, 0.5],  # Sete Rios
        [0, 0, 0, 0, 1],  # Praça de Londres
        [1, 0, 0, 0, 0],  # Alameda
    ])
    return state_space, probability_space


def prob_trajectory(markov_chain, sequence):
    state_space, transition_matrix = markov_chain[0], markov_chain[1]
    probability = 1
    for i in range(len(sequence) - 1):
        probability *= transition_matrix[state_space.index(sequence[i])][state_space.index(sequence[i + 1])]
    return probability


def stationary_dist(markov_chain):
    values, left, right = linalg.eig(markov_chain[1], right=True, left=True)
    left_eigenvector = left[:, 0]
    left_eigenvector = -np.linalg.eig(markov_chain[1].T)[1][:, 0]
    return left_eigenvector / linalg.norm(left_eigenvector, ord=1)


def compute_dist(markov_chain, distribution, number_of_steps):
    return np.dot(distribution, np.linalg.matrix_power(markov_chain[1], number_of_steps))


def simulate(markov_chain, distribution, number_of_steps):
    state_space, transition_matrix = markov_chain[0], markov_chain[1]
    sequence = []

    for i in range(number_of_steps - 1):
        # check max index
        max_index = np.argmax(distribution)
        # append state with max index
        sequence.append(state_space[max_index])
        # update distribution
        distribution = np.zeros(len(state_space))
        distribution[max_index] = 1

        distribution = np.dot(distribution, transition_matrix)

    return tuple(sequence)



data = np.array(traj)
plt.hist(x = data, align = 'mid')
plt.show()
