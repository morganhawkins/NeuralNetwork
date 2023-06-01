import numpy as np

#function to do nothing -- helps clean code use din debugging
def silent(a = None):
    return


#shuffles vectors in unison
def shuffle_together(v_1, v_2):
    assert len(v_1) == len(v_2), "vectors different length"
    
    perm = np.random.permutation(len(v_1))
    return v_1[perm], v_2[perm]