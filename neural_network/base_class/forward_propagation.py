# neural_network/base_class/forward_propagation.py
import numpy as np


class Activation:
    
    def sigmoid(self, Z):
        """
        Implements the sigmoid activation in numpy
        
        Arguments:
        Z -- numpy array of any shape
        
        Returns:
        A -- output of sigmoid(z), same shape as Z
        cache -- returns Z as well, useful during backpropagation
        """
        
        A = 1/(1+np.exp(-Z))
        cache = Z
        
        return A, cache

    def relu(self, Z):
        """
        Implement the RELU function.

        Arguments:
        Z -- Output of the linear layer, of any shape

        Returns:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
        """
        
        A = np.maximum(0,Z)
        
        assert(A.shape == Z.shape)
        
        cache = Z 
        return A, cache

class ForwardPropagation(Activation):
    ### 2.1 Implementación de linear forward ###
    def linear_forward(self, A: np.array, W: np.array, b: np.array) -> tuple:
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
        
        Z = np.dot(W, A) + b
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache

    ### 2.2 Implementación de linear-activation forward ###
    def linear_activation_forward(self, A_prev: np.array, W: np.array, b: np.array, activation: str) -> tuple:
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                stored for computing the backward pass efficiently
        """
        
        # test for valid options
        assert activation == "sigmoid" or activation == "relu"
        
        # test for object type
        assert isinstance(A_prev, np.ndarray), "A_prev no es un np.ndarray"
        assert isinstance(W, np.ndarray), "W no es un np.ndarray"
        assert isinstance(b, np.ndarray), "b no es un np.ndarray"
        
        # test for shape
        assert A_prev.ndim == 2, "A_prev no es un rank 2 array"
        assert W.ndim == 2, "W no es un rank 2 array"
        assert b.ndim == 2, "b no es un rank 2 array"
        
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)        
        
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
        cache = (linear_cache, activation_cache)

        return A, cache

    ###   2.3) Combinación de 2.1 y 2.2  ###
    def L_model_forward(self, X: np.array, parameters: dict) -> tuple:
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- activation value from the output (last) layer
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2 #  number of layers in the neural network
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(
                A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu'
                )
            caches.append(cache)        
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL , cache = self.linear_activation_forward(
            A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid'
            )
        caches.append(cache)    
            
        return AL, caches

