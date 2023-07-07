
import numpy as np
from nnlearn.Helper_Functions import *
from numba import njit

class connected_layer:
    def __init__(self, num_neurons, prev_neurons):
        
        #number of neurons in layer and previous layer
        self.size = num_neurons 
        self.prev_size = prev_neurons 
        
        #weight matrix and bias matrix initialization
        
        #-must randomize weight matrix so that gradient isn't same for all neurons
        #--because gradient of neuron w respect to previous neuron is just associated weight
        self.weight_mat = np.random.uniform(-1, 1, (num_neurons, prev_neurons))
        self.weight_mat_update = np.zeros((num_neurons, prev_neurons))
        
        
        #-want to be column vector
        #--no reason to not initialize to 0
        self.bias_mat = np.zeros((num_neurons, 1))
        self.bias_mat_update = np.zeros((num_neurons, 1))
        
        self.params = num_neurons  * (prev_neurons + 1)
    
    #pushes sample through layer and remembers output
    #also calcualtes gradient to previous layer
    def forward(self, activations):
        self.out = self.weight_mat @ activations + self.bias_mat
        
        self.gradient_to_prev = self.gradient()
        
        return self.out

    
    #returns gradient of layer output (vector)  with respect to weights (matrix)
    #loose definition check notes for how this is defined
    def gradient(self):
        return self.weight_mat.T

    
class activation_layer:
    def __init__(self, size):
        
        self.size = size
        self.prev_size = size
        self.params = 0
    
    
class relu_activation_layer(activation_layer):

    def gradient(self):
        return np.diag(d_relu(self.out).flatten())
    

    def forward(self, x):
        assert x.shape == (self.size, 1), f"input and layer size incompatible, {x.shape} passed"
        
        self.out = relu(x)
        self.gradient_to_prev = self.gradient()
        
        return self.out
    


class sigmoid_activation_layer(activation_layer):
    
    def gradient(self):

        flat_out = self.out.flatten()

        return np.diag(flat_out*(1-flat_out))
    

    def forward(self, x):
        assert x.shape == (self.size, 1), f"input and layer size incompatible, {x.shape} passed"
        
        self.out = sigmoid(x)
        self.gradient_to_prev = self.gradient()
        
        return self.out

    

    
class input_layer:
    def __init__(self, size):
        self.size = size
        self.prev_size = None
        self.gradient_to_prev = None
        self.params = 0
        
    def gradient(self):
        return None
    
    #feeds sample to first fully connected layer
    def forward(self, x):
        assert len(x) == self.size, "input incorrect length" 
        
        self.out = x
        
        return x
        
        