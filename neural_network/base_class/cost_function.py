# neural_network/base_class/cost_function.py
import numpy as np


### 3) Función de costo  ###
### Para implementar la función de costo necesitamos un test set para comparar el valor de las predicciones
class CostFunction:

    def compute_cost(self, AL: np.array, Y: np.array) -> float:
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        
        m = Y.shape[1]

        cost = -np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))/m    
        
        cost = np.squeeze(cost) #  To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

        return cost
