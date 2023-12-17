# neural_network/snn_model.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neural_network.base import NeuralNetworkBase

class ShallowNNClassifier(BaseEstimator, TransformerMixin, NeuralNetworkBase):
    def __init__(self, X_test, Y_test, learning_rate=0.1, num_iterations=400, print_cost=False):
        self.X_test = X_test
        self.Y_test = Y_test
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
        
        self.X_flatten_shape = train_x_flatten.shape
        self.X_test_flatten_shape = test_x_flatten.shape
        
        # Hyperparameters
        n_x = X_.shape[1] # n input features
        n_h = n_x # neuronas in the hidden layer
        n_y = 1 # n neurons in the output layer
        
        # Training
        self.parameters, self.costs = self.two_layer_model(
            train_x_flatten, Y_, 
            layers_dims = (n_x, n_h, n_y), 
            num_iterations = self.num_iterations,
            learning_rate = self.learning_rate,
            print_cost = self.print_cost
            )
        
        return self
    
    def feed_exploration(self, X, Y, X_test, Y_test):#DONE
        # Convert to column vectors
        X = X.to_numpy()
        Y = Y.to_numpy().reshape(Y.shape[1], Y.shape[0])
        X_test = X_test.to_numpy()
        Y_test = Y_test.to_numpy().reshape(Y_test.shape[1], Y_test.shape[0])
        
        self.Y_shape = Y.shape
        self.Y_test_shape = Y_test.shape
        
        print ("Number of training examples: " + str(X.shape[0]))
        print ("Number of testing examples: " + str(X_test.shape[0]))
        print ("Training set shape: " + str(X.shape))
        print ("Training vector shape: " + str(Y.shape))
        print ("Testing set shape: " + str(X_test.shape))
        print ("Testing vector shape: " + str(Y_test.shape))

        return X, Y, X_test, Y_test
    
    def resample_X(self, X):#DONE
        X_ = X.copy()
        if isinstance(X_, type(pd.DataFrame())):
            X_ = np.array(X_.to_numpy(), dtype=np.float64)
        X_flatten = X_.reshape(X_.shape[0], -1).T
        
        assert X_flatten.shape[0] == self.X_flatten_shape[0], "X_flatten shape must be equal to X_flatten_shape"
        return X_flatten
    
    def resample_Y(self, Y):#DONE
        Y_ = Y.copy()
        if isinstance(Y_, pd.core.series.Series):
            Y_ = np.array(Y_.to_numpy(), dtype=np.float64)
        Y_flatten = Y_.reshape(Y_.shape[0], -1).T
        
        assert Y_flatten.shape[0] == self.Y_shape[0], "Y_flatten shape must be equal to Y_shape"
        return Y_flatten    
    
    def two_layer_model(self, X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
        """
        Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
        
        Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
        layers_dims -- dimensions of the layers (n_x, n_h, n_y)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- If set to True, this will print the cost every 100 iterations 
        
        Returns:
        parameters -- a dictionary containing W1, W2, b1, and b2
        """
        
        np.random.seed(1)
        grads = {}
        costs = []                              # to keep track of the cost
        m = X.shape[1]                           # number of examples
        (n_x, n_h, n_y) = layers_dims
        
        parameters = self.initialize_parameters(n_x, n_h, n_y)
        
        # Get W1, b1, W2 and b2 from the dictionary parameters.
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
            A1, cache1 = self.linear_activation_forward(X, W1, b1, 'relu')
            A2, cache2 = self.linear_activation_forward(A1, W2, b2, 'sigmoid')
    
            # Compute cost
            cost = self.compute_cost(A2, Y)        
            
            # Initializing backward propagation
            dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
            
            # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
            dA1, dW2, db2 = self.linear_activation_backward(dA2, cache2, 'sigmoid')
            dA0, dW1, db1 = self.linear_activation_backward(dA1, cache1, 'relu')
            
            # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
            grads['dW1'] = dW1
            grads['db1'] = db1
            grads['dW2'] = dW2
            grads['db2'] = db2
            
            # Update parameters.
            parameters = self.update_parameters(parameters, grads, learning_rate)

            # Retrieve W1, b1, W2, b2 from parameters
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
            
            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {:.4f}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)

        return parameters, costs
    
    def predict(self, X, y, threshold=0.5):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        if X.shape[0] != self.X_flatten_shape[0]:
            X = self.resample_X(X)
        if y.shape[0] != self.Y_shape[0]:
            y = self.resample_Y(y)
        
        m = X.shape[1]
        n = len(self.parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.L_model_forward(X, self.parameters)

        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > threshold:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        print("Accuracy: {:.4f}".format(np.sum((p == y)/m)))
            
        return p

    def predict_proba(self, X):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        if X.shape[0] != self.X_flatten_shape[0]:
            X = self.resample_X(X)
        
        # Forward propagation
        probas, caches = self.L_model_forward(X, self.parameters)
            
        return probas
    

    def plot_costs(self, y_min=None, y_max=None):
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.learning_rate))
        
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        
        plt.show()
        
    def plot_proba_dist(self, X):
        probas = self.predict_proba(X)
        plt.hist(probas[0], bins=255, density=True)
        plt.title("Probability distribution")
        plt.xlabel("Probability")
        plt.ylabel("Frequency")
        plt.show()
        
    def transform(self, X):
        result = self.predict_proba(X)
        result = pd.DataFrame(result, columns=["snn_proba"])
        return result

