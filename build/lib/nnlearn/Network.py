import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit as sigmoid

from time import time
from random import sample



#importing other files
from nnlearn.Layers import *
from nnlearn.Helper_Functions import *

#Creating NN object

class network:
    def __init__(self, layers):

        self.layers = layers
        self.num_layers = len(layers)
        
        #input size = number of weights associated w/ each neuron in first layer
        self.input_size = layers[0].size
        self.output_size = layers[-1].size
        
        self.epochs_trained = 0
        self.loss_history = np.empty(0)
        self.train_time = 0
        
        self.verify()
    
    def reset_update_mats(self):
        """
        sets the parameter update mats for each layers to 0
        """

        for layer in self.layers:
            if type(layer) == connected_layer:
                layer.weight_mat_update = abs(layer.weight_mat_update*0)
                layer.bias_mat_update = abs(layer.bias_mat_update*0)
                

    def loss(self, x, y):
        assert x.shape[0] == y.shape[0], "x and y incompatible shapes"
        
        squared_error = 0
        for i in range(len(x)):
            squared_error += sum((self.predict(x[i]).flatten() - y[i])**2)
        
        return squared_error/len(y.flatten())
    
    def graph_loss_history(self, yscale = "log"):
        """
        plots the train loss against epcohs
        provides info on train speed
        """

        assert yscale in ["log", "linear"], "invalid yscale passed"
        
        plt.figure(figsize = (7,4))
        plt.style.use("ggplot")
        plt.plot(np.arange(self.epochs_trained), self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel(yscale + " Loss")
        plt.title("Training Loss")

        y_range = max(self.loss_history) - min(self.loss_history)

        plt.text(self.epochs_trained*.7, y_range*.95 + min(self.loss_history),
                 f"ms/epoch: {round(self.train_time/self.epochs_trained*1000,3)}",
                 color = "darkgreen",)
        
        plt.text(self.epochs_trained*.7, y_range*.70 + min(self.loss_history),
                 f" train loss: {round(self.train_loss,4)}",
                 color = "darkgreen",)
        
        plt.yscale(yscale)
        
        plt.show()
                
        
    def verify(self):
        """
        verifies that the network's layers match input/output shapes

        """

        for i in range(1,len(self.layers)):
            assert self.layers[i-1].size == self.layers[i].prev_size, "invalid network"
            
            
    def verify_sample(self, x, y):
        """
        verifies that a sample is valid for the network

        """

        assert len(x) == len(y), "x and y havent different numebr of samples"
        
        for i in range(len(x)):
            assert len(x[i]) == self.input_size, f"sample at index [{i}] incorrect input size"
            assert len(y[i]) == self.output_size, f"sample at index [{i}] incorrect output size"
        
        
    def predict(self, inp):
        """
        pushes sample through network and return output of final layer

        """
        inp = inp.reshape(self.input_size,1)
        assert inp.shape == (self.input_size, 1), "wrong input size"
        
        for i in range(self.num_layers):

            inp = self.layers[i].forward(inp)
            
        return inp
    
    
    def print_structure(self):
        """
        prints the structure of the network and number of parameters

        """
        total_params = 0
        for i,layer in enumerate(self.layers[::-1]):
            print("-"*45)
            print(type(layer))
            print(f"  layer: {self.num_layers-i-1}/{self.num_layers-1}")
            print(f"neurons: {layer.size}")
            print(f" prev n: {layer.prev_size}")
            print(f" params: {layer.params}")
            print()
            total_params += layer.params
            
        print("-" * 45, f"\ntotal params: {total_params}")
        
   
        
    def forward(self, x):
        """
        pushes sample through the network, updating the "out" attributes of each layer and it's gradient to it's parameters

        """


        assert x.shape == (self.input_size,1)
        
        for layer in self.layers:
            x = layer.forward(x)
            
            
    def backward(self, y, learn_coef = 1):
        """
        calculates the "weight_mat_update" and "bias_mat_update" attributes of each layer
        
        """

        g_cost_layer = np.array([2*(self.layers[-1].out[i] - y[i]) for i in range(self.output_size)])
        
        
        #Looping through layers: going from end --> start 
        
        i = len(self.layers) - 1
        
        while i > 0:
             
            #if it's a connected layer we want to update its weights and biases
            if type(self.layers[i]) == connected_layer:
                
                #updating weights

                g_layer_weights = np.array([self.layers[i-1].out.flatten() for c in range(self.layers[i].size)])
                
                g_cost_weights = np.diag(g_cost_layer.flatten()) @ g_layer_weights
                
            
                self.layers[i].weight_mat_update = self.layers[i].weight_mat_update - (g_cost_weights*learn_coef)
                
                
                #updating biasses
                
                g_cost_bias = g_cost_layer
                
                self.layers[i].bias_mat_update = self.layers[i].bias_mat_update - (g_cost_bias*learn_coef)
                
                
            #updating cost to layer nodes gradient
            g_cost_layer = self.layers[i].gradient_to_prev @ g_cost_layer
            
 
            
            i -= 1
        
        
    def minibatch_fit(self, x, y, batch_size = None, epochs = 10, learn_coef = .1, verbose = True):
        
        """
        Function to train the network through minibatch SGD. 
        
        x: np array of size (n,p) - input array
        y: np array of size (n,o) - output array
        batch_size: int in [0,n]  - number of samples used to calculate gradient at each epoch
        epcohs: int in [1, inf)   - number of epochs used in training
        learn_coef: doub in [0,1] - coeficient to adjust gradient by
        verbose: T,F              - whether to print output

        """

        self.reset_update_mats()

        assert (type(x) == np.ndarray) and (type(y) == np.ndarray), "X and Y must both be numpy arrays"

        assert x.shape[0] == y.shape[0], f"X and Y have incomaptible shapes {x.shape} {y.shape}"

        assert (learn_coef >= 0) and (learn_coef <= 1), f"learn coef must be in [0,1]. learn_coef passed: {learn_coef}"

        if batch_size == None: batch_size = x.shape[0] 


        loss_history = np.zeros(epochs)

        start_time = time()

        for epoch in range(epochs):

            epoch_start = time()
            
            check_start = time()

            batch_indices = sample(range(len(x)), batch_size)

            # check_end = time()
            # print(f"sample:{check_end - check_start}")
            # check_start = time()


            for i,s in enumerate(batch_indices):
                self.forward(x[s].reshape(-1,1))
                self.backward(y[s].reshape(-1,1), learn_coef)

                # check_end = time()
                # print(f"batch {i} fb:{check_end - check_start}")
                # check_start = time()

            for layer in self.layers: 
                if type(layer) == connected_layer:
                    layer.weight_mat = layer.weight_mat + (layer.weight_mat_update/len(x))
                    layer.bias_mat = layer.bias_mat + (layer.bias_mat_update/len(x))

            # check_end = time()
            # print(f"update params:{check_end - check_start}")
            # check_start = time()

            self.reset_update_mats()

            # check_end = time()
            # print(f"reset update mats:{check_end - check_start}")
            # check_start = time()


            
            self.epochs_trained += 1

            if (verbose) and ((epoch%(epochs/5) == 0) or (epoch == epochs - 1)):
                loss = self.loss(x, y)
                print("-"*20)     
                print(f"epoch: {self.epochs_trained} \n loss: {round(loss, 4)}")
            else:
                loss = 0


            loss_history[epoch] = loss
            

            # check_end = time()
            # print(f"collect data: {check_end - check_start}")
            # check_start = time()


            # print(f"full epoch: {time() - epoch_start}")
            # print("\n" , "-"*45, "\n")
                
            

        self.train_time += time() - start_time
        self.train_loss = self.loss(x, y)
        self.loss_history = np.concatenate((self.loss_history, loss_history))




        
            



