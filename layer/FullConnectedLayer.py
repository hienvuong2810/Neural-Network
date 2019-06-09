#!/usr/bin/env python
# coding: utf-8

# In[1]:


from .layer import Layer
import numpy as np
class FullConnectedLayer(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights  = np.random.rand(input_shape[1], output_shape[1]) - 0.5
        self.bias = np.random.rand(1,output_shape[1]) - 0.5
        print(self.weights)
    def forward_propagation(self, input):
        self.input  = input
        self.output = np.dot(self.input, self.weights)
        return self.output + self.bias

    def backward_propagation (self, output_error, learning_rate):
        curr_layer_error = np.dot(output_error, self.weights.T)
        dweights = np.dot(self.input.T, output_error)
        self.weights -= dweights * learning_rate
        self.bias -= learning_rate * output_error

        return curr_layer_error


# In[ ]:
