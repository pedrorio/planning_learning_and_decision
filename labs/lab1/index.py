import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(42)


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
    # values, left, right = linalg.eig(markov_chain[1], right=True, left=True)
    # left_eigenvector = left[:, 0]
    left_eigenvector = -np.linalg.eig(markov_chain[1].T)[1][:, 0]
    return left_eigenvector / np.linalg.norm(left_eigenvector, ord=1)


def compute_dist(markov_chain, distribution, number_of_steps):
    return np.dot(distribution, np.linalg.matrix_power(markov_chain[1], number_of_steps))


def simulate(markov_chain, distribution, number_of_steps):
    state_space, transition_matrix = markov_chain
    state_space = np.asanyarray(state_space)
    distribution = distribution[0]
    sequence = []
    for i in range(number_of_steps):
        selected_state = np.random.choice(state_space, p=distribution)
        sequence.append(selected_state)
        distribution = transition_matrix[np.where(state_space == selected_state)[0][0], :]
    return tuple(sequence)


def create_plot():
    M = load_chain()
    nS = len(M[0])
    u = np.ones((1, nS)) / nS

    plt.hist(x=list(map(int, simulate(M, u, 10000))), bins=[0, 1, 2, 3, 4, 5], align='left', rwidth=0.5)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    plt.show()

    return Counter(simulate(M, u, 10000))