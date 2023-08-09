import numpy as np
from numba import njit

from scipy.special import expit as sigmoid


def silent(a = None):
    """
    does nothing - useful for debugging sometimes
    """

    return



def shuffle_together(v_1, v_2):
    """
    shuffles two vectors in unison
    """

    assert len(v_1) == len(v_2), "vectors different length"
    
    perm = np.random.permutation(len(v_1))
    return v_1[perm], v_2[perm]


@njit
def relu(x):
    """
    calcualates output of relu layer given values of previous layer

    """

    return np.maximum(0,x)


@njit
def d_relu(output):
    """
    calcualtes gradient of relu layer nodes with respect to its previous nodes

    """

    return 1 * (output > 0)



@njit
def leaky_relu(x):
    """
    calcualates output of leaky relu layer given values of previous layer
    """

    return np.where(x > 0, x, 0.01*x)    


@njit
def d_leaky_relu(output):
    """
    calcualtes gradient of leaky relu layer nodes with respect to its previous nodes
    """

    return np.where(output > 0, 1, 0.01)     