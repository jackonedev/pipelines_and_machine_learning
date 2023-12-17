import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from base_class.initialization import Initialization
from base_class.forward_propagation import ForwardPropagation
from base_class.cost_function import CostFunction
from base_class.backward_propagation import BackwardPropagation
from base_class.update_parameters import UpdateParameters

class NeuralNetworkBase(Initialization, ForwardPropagation, CostFunction, BackwardPropagation, UpdateParameters):
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
    
    def resample_X(self, X):
        X_ = X.copy()
        if isinstance(X_, type(pd.DataFrame())):
            X_ = np.array(X_.to_numpy(), dtype=np.float64)
        X_flatten = X_.reshape(X_.shape[0], -1).T
        
        assert X_flatten.shape[0] == self.X_flatten_shape[0], "X_flatten shape must be equal to X_flatten_shape"
        return X_flatten
    
    def resample_Y(self, Y):
        Y_ = Y.copy()
        if isinstance(Y_, pd.core.series.Series):
            Y_ = np.array(Y_.to_numpy(), dtype=np.float64)
        Y_flatten = Y_.reshape(Y_.shape[0], -1).T
        
        assert Y_flatten.shape[0] == self.Y_shape[0], "Y_flatten shape must be equal to Y_shape"
        return Y_flatten
    
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