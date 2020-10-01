import numpy as np
from labs.lab1.index import load_chain, prob_trajectory, stationary_dist


class TestLab1:
    MARKOV_CHAIN = (
        ('0', '1', '2', '3', '4'),
        np.array([
            [0, 0.3, 0.7, 0, 0],  # Tagus
            [0, 0, 1, 0, 0],  # Oeiras
            [0, 0, 0, 0.5, 0.5],  # Sete Rios
            [0, 0, 0, 0, 1],  # Praça de Londres
            [1, 0, 0, 0, 0],  # Alameda
        ])
    )

    STATIONARY_DISTRIBUTION = np.array([0.263, 0.079, 0.263, 0.132, 0.263])

    class TestLoadChain:
        def test_result_is_tuple(self):
            assert type(TestLab1.MARKOV_CHAIN) is tuple

        def test_state_space_is_tuple(self):
            assert type(TestLab1.MARKOV_CHAIN[0]) is tuple

        def test_probability_matrix_is_np_array(self):
            assert type(TestLab1.MARKOV_CHAIN[1]) is np.ndarray

        def test_state_space_result(self):
            assert load_chain()[0] == TestLab1.MARKOV_CHAIN[0]

        def test_probability_matrix_result(self):
            assert (load_chain()[1] == TestLab1.MARKOV_CHAIN[1]).all()

        def test_state_space_length(self):
            assert len(TestLab1.MARKOV_CHAIN[0]) == 5

        def test_probability_matrix_shape(self):
            assert TestLab1.MARKOV_CHAIN[1].shape == (5, 5)

    class TestProbTrajectory:

        def test_first_trajectory(self):
            assert prob_trajectory(TestLab1.MARKOV_CHAIN, ('1', '2', '3')) == 0.5

        def test_second_trajectory(self):
            assert prob_trajectory(TestLab1.MARKOV_CHAIN, ('4', '0', '1', '2', '3')) == 0.15

        def test_third_trajectory(self):
            assert prob_trajectory(TestLab1.MARKOV_CHAIN, ('4', '0', '4')) == 0.0

        def test_fourth_trajectory(self):
            assert prob_trajectory(TestLab1.MARKOV_CHAIN, ('0', '2', '4')) == 0.35

    class TestStationaryDist:

        def test_sums_to_one(self):
            assert np.sum(stationary_dist(TestLab1.MARKOV_CHAIN)) == 1

        def test_is_close(self):
            np.all(np.isclose(stationary_dist(TestLab1.MARKOV_CHAIN), TestLab1.STATIONARY_DISTRIBUTION))
