import numpy as np
import matplotlib.pyplot as plt
from numba import njit

class loss_function:

    def __init__(self):
        pass


class mse_loss(loss_function):

    def gradient(self, prediction, target):
        return 2 * (prediction - target)
        
    def loss(self, prediction, target):
        return np.array((prediction - target)**2).mean()



class cross_entropy_loss(loss_function):

    def gradient(self, prediction, target):

        assert prediction.shape == target.shape, f"prediction & target shape different {prediction.shape} {target.shape}"
        assert len(prediction.flatten()) == len(prediction), "make sure only using for binary cross entropy"

        # prediction = prediction.flatten()
        # target = target.flatten()

        prediction = np.where(prediction == 0, .0001, prediction)
        prediction = np.where(prediction == 1, .9999, prediction)

        return ((1 - target)/(1-prediction)) - (target / prediction )
        
        
    def loss(self, prediction, target):

        assert prediction.shape == target.shape, f"prediction & target shape different {prediction.shape} {target.shape}"
        assert len(prediction.flatten()) == len(prediction), "make sure only using for binary cross entropy"

        
        prediction = prediction.flatten()
        target = target.flatten()

        prediction = np.where(prediction == 0, .0001, prediction)
        prediction = np.where(prediction == 1, .9999, prediction)

        return -np.array((target * np.log(prediction)) + ((1-target)*np.log(1-prediction))).mean()
        


# loss_func = mse_loss(5)

# vec_1 = np.array([.1, .9, .9, .1])
# vec_2 = np.array([0, 0, 1, 1])

# vec_1 = np.random.uniform(size = 4)
# vec_2 = np.round(np.random.uniform(size = 4))

# print(vec_1, vec_2, sep = "\n")

# print("\n",loss_func.loss(vec_1, vec_2))

# print("\n", -loss_func.gradient(vec_1, vec_2))


