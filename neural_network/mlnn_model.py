# neural_network/mlnn_model.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neural_network.base import NeuralNetworkBase

class MultiLayerNNClassifier(BaseEstimator, TransformerMixin, NeuralNetworkBase):
    def __init__(self, X_test, Y_test, layers_dim, learning_rate=0.1, num_iterations=400, print_cost=False):
        self.X_test = X_test
        self.Y_test = Y_test
        self.layers_dim = layers_dim
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.model_params = None
        self.model_cost = None
        self.print_cost = print_cost
        
    def fit(self, X, y=None):
        assert self.num_iterations > 0, "num_iterations must be greater than 0"
        assert self.learning_rate > 0, "learning_rate must be greater than 0"
        assert isinstance(X, type(pd.DataFrame())), "Type of X must be pandas.DataFrame"
        assert isinstance(y, pd.core.series.Series), "Type of y must be pandas.Series"
        self.X_ = X.copy()
        self.Y_= y.copy()
        
        X_, Y_, X_test, Y_test = self.feed_exploration(
            X, y.to_frame(), self.X_test, self.Y_test.to_frame())
        
        train_x_flatten = np.array(X_, dtype=np.float64)
        test_x_flatten = np.array(X_test, dtype=np.float64)
        train_x_flatten = train_x_flatten.reshape(train_x_flatten.shape[0], -1).T
        test_x_flatten = test_x_flatten.reshape(test_x_flatten.shape[0], -1).T
        
        # train_x_flatten = normalize_rows(train_x_flatten)
        # test_x_flatten = normalize_rows(test_x_flatten)
        # train_x_flatten = softmax(train_x_flatten)
        # test_x_flatten = softmax(test_x_flatten)
        
        self.X_flatten_shape = train_x_flatten.shape
        self.X_test_flatten_shape = test_x_flatten.shape
        
        # Training
        self.parameters, self.costs = self.L_layer_model(
            train_x_flatten, Y_, 
            layers_dims = self.layers_dim,#TODO: crear test para layers_dim 
            num_iterations = self.num_iterations,
            learning_rate = self.learning_rate,
            print_cost = self.print_cost
            )
        
        return self
        
    def L_layer_model(self, X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        
        Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = []                         # keep track of cost
        
        parameters = self.initialize_parameters_deep(layers_dims)    
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self.L_model_forward(X, parameters)
        
            # Compute cost.
            cost = self.compute_cost(AL, Y)        
        
            # Backward propagation.
            grads = self.L_model_backward(AL, Y, caches)
    
            # Update parameters.
            parameters = self.update_parameters(parameters, grads, learning_rate)        
                    
            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)
        
        return parameters, costs

    def transform(self, X):
        result = self.predict_proba(X)
        result = pd.DataFrame(result, columns=["mlnn_proba"])
        return result
