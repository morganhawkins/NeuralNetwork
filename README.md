# Overview

This library serves as a framework for creating artificial neural networks from a variety of connected and activation layers. Everything primarily utilizes the numpy library. No pytorch or tensorflow is used, but functions are compiled when possibloe using the numba library. This project was created for my understanding of neural networks.

### Network Classes

- Network: artificial neural network consisting of layer objects passed at initialization
- Compresser: neural network that automatically tranforms images into tabular data for training and tranforms predictions into images for recreation

### Layer Classes

- Input Layer: accepts sample into network
- Connected Layer: fully connected layer with no activation
- Activation Layer: adds activations to previous layer (Relu, Sigmoid, and Leaky ReLu)

### Training Methods
- Gradient Decent
- Stochastic/ Minibatch Gradient Decent



# Visualizations

### Training a network to reproduce the digit 3.


This is the original image that we would like our network to be able to transform into an image of higher resolution.

&emsp;&emsp; Original

<img src="https://github.com/morganhawkins/NeuralNetwork/blob/main/images/digit_3.png " width="150" height="150" />


All three images show identical networks with different activation functions. 

&emsp;&emsp; Leaky ReLu &emsp;&emsp;&emsp;&emsp;&nbsp;  Sigmoid  &emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;    Mixed

<img src="https://github.com/morganhawkins/NeuralNetwork/blob/main/images/leaky_relu_recreation_looping.gif " width="150" height="150" /><img src="https://github.com/morganhawkins/NeuralNetwork/blob/main/images/sigmoid_receation_looping.gif " width="150" height="150" /><img src="https://github.com/morganhawkins/NeuralNetwork/blob/main/images/mixed_recreation_looping.gif " width="150" height="150" />









