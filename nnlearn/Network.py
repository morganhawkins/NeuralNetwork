import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit as sigmoid
from time import time
from time import sleep

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
        
        plt.text(self.epochs_trained*.7, y_range*.75 + min(self.loss_history),
                 f" train loss: {round(self.train_loss,4)}",
                 color = "darkgreen",)
        
        plt.yscale(yscale)
        
        plt.show()
                
        
    def verify(self):
        for i in range(1,len(self.layers)):
            assert self.layers[i-1].size == self.layers[i].prev_size, "invalid network"
            
            
    def verify_sample(self, x, y):
        assert len(x) == len(y), "x and y havent different numebr of samples"
        
        for i in range(len(x)):
            assert len(x[i]) == self.input_size, f"sample at index [{i}] incorrect input size"
            assert len(y[i]) == self.output_size, f"sample at index [{i}] incorrect output size"
        
        
    def predict(self, inp):
        inp = inp.reshape(self.input_size,1)
        assert inp.shape == (self.input_size, 1), "wrong input size"
        
        for i in range(self.num_layers):

            inp = self.layers[i].forward(inp)
            
        return inp
    
    
    def print_structure(self):
        for i,layer in enumerate(self.layers[::-1]):
            print("-"*45)
            print(type(layer))
            print(f"  layer: {self.num_layers-i-1}/{self.num_layers-1}")
            print(f"neurons: {layer.size}")
            print(f" prev n: {layer.prev_size}")
            print()
   
        
    def forward(self, x):
        assert x.shape == (self.input_size,1)
        
        for layer in self.layers:
            x = layer.forward(x)
            
            
    def backward(self, y, learn_coef = 1):
        
        g_cost_layer = np.array([2*(self.layers[-1].out[i] - y[i]) for i in range(self.output_size)])
        
        
        #Looping through layers: going from end --> start 
        
        i = len(self.layers)-1
        
        while i>0:
             
            #if it's a connected layer we want to update its weights and biases
            if type(self.layers[i]) == connected_layer:
                
                #updating weights

                g_layer_weights = np.array([self.layers[i-1].out.flatten() for c in range(self.layers[i].size)])
                
                g_cost_weights = np.diag(g_cost_layer.flatten()) @ g_layer_weights
                
            
                self.layers[i].weight_mat_update = self.layers[i].weight_mat_update - (g_cost_weights*learn_coef)
                
                
                #updating biasses
                
                g_cost_bias = g_cost_layer
                
                self.layers[i].bias_mat_update = self.layers[i].bias_mat_update - (g_cost_bias*learn_coef)
                
                
            
            try:
            
                g_cost_layer = self.layers[i].gradient_to_prev @ g_cost_layer
            
            except:
                print(f'backwards -- layer: {i}')
                
                print(self.layers[i].gradient_to_prev)
            
                print(g_cost_layer)
            
            i -= 1
        
        
    def fit(self, x, y, epochs = 10, learn_coef = .1, verbose = True):

        
        loss_history = np.zeros(epochs)

        start_time = time()
        for epoch in range(epochs):

            for s in range(len(x)):
                self.forward(x[s].reshape(-1,1))
                self.backward(y[s].reshape(-1,1), learn_coef)

            for layer in self.layers: 
                if type(layer) == connected_layer:
                    layer.weight_mat = layer.weight_mat + (layer.weight_mat_update/len(x))
                    layer.bias_mat = layer.bias_mat + (layer.bias_mat_update/len(x))

            self.reset_update_mats()

            loss = self.loss(x, y)
            loss_history[epoch] = loss
            self.epochs_trained += 1

            if (verbose) and ((epoch%(epochs/5) == 0) or (epoch == epochs - 1)):
                print("-"*20)     
                print(f"epoch: {self.epochs_trained} \n loss: {round(loss, 4)}")
                
            

        self.train_time += time() - start_time
        self.train_loss = self.loss(x, y)
        self.loss_history = np.concatenate((self.loss_history, loss_history))
        
        
            



