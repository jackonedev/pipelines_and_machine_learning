# neural_network/base_model/initialization.py
import numpy as np

class Initialization:
        
    ###  1.1 Inicializacion red de dos capas (1 hidden, 1 output) ###
    def initialize_parameters(self, n_x: int, n_h: int, n_y: int) -> dict:
        """

        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        
        Returns:
        parameters -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """
        
        np.random.seed(1)
        
        W1 = np.random.randn(n_h, n_x)*0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h)*0.01
        b2 = np.zeros((n_y, 1))    
        
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        
        return parameters 

    ###  1.2 Inicializacion red de L capas ###
    def initialize_parameters_deep(self, layer_dims: list) -> dict:
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims) # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

            
        return parameters
