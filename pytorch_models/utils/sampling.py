import numpy as np


def random_choice_prob_index_2d(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


def choice2d(a, axis=0):
    rng = np.random.default_rng()
    return rng.choice(a, axis=axis)
