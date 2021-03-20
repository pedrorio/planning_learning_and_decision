import numpy as np
from lab2.lab2 import load_mdp, noisy_policy


class TestLab2:
    class TestLoadMDP:
        def test_number_of_states(self):
            X, *_ = load_mdp('labs/lab2/maze.npz', 0.9)
            assert len(X) == 73

        def test_last_state(self):
            X, *_ = load_mdp('labs/lab2/maze.npz', 0.9)
            assert X[-1] == "E"

        def test_number_of_actions(self):
            _, A, *_ = load_mdp('labs/lab2/maze.npz', 0.9)
            assert len(A) == 4

        def test_gamma(self):
            *_, gamma = load_mdp('labs/lab2/maze.npz', 0.9)
            assert gamma == 0.9

    class TestNoisyPolicy:
        def test_policy_with_index_2(self):
            pol_noisy = noisy_policy(load_mdp('labs/lab2/maze.npz', 0.9), 2, 0.75)
            assert np.all(pol_noisy[14, :] == np.array([0.03, 0.03, 0.9,  0.03]))
