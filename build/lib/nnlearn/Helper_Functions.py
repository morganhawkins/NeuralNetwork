import numpy as np
from numba import njit


from scipy.special import expit as sigmoid

#function to do nothing -- helps clean code use din debugging
def silent(a = None):
    return


#shuffles vectors in unison
def shuffle_together(v_1, v_2):
    assert len(v_1) == len(v_2), "vectors different length"
    
    perm = np.random.permutation(len(v_1))
    return v_1[perm], v_2[perm]


@njit
def relu(x):
    return np.maximum(0,x)


@njit
def d_relu(output):
    return 1 * (output > 0)



@njit
def leaky_relu(x):
    return np.where(x > 0, x, 0.01*x)    


@njit
def d_leaky_relu(output):
    return np.where(output > 0, 1, 0.01)     