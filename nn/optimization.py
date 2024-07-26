from typing import Tuple, Dict, List
from autograd import grad
import numpy as np
from abc import ABC, abstractmethod

from autograd.array import Array



class BaseOptimizer(ABC):

    def __init__(self, parameter_arrays: List[Array], lr: float, l2_reg_coeff: float = 0.0) -> None:
        """ Base Optimization class

        Args:
            parameter_arrays (List[Array]): List of parameter Arrays
            lr (float): Learning rate
            l2_reg_coeff (float, optional): L2 regularization coefficient. Defaults to 0.0.
        """
        self.parameter_arrays = parameter_arrays
        self.lr = lr
        self.l2_reg_coeff = l2_reg_coeff

    @abstractmethod
    def update(self, gradients: Dict[Array, np.ndarray]) -> None:
        pass


class SGD(BaseOptimizer):

    def update(self, gradients: Dict[Array, np.ndarray]) -> None:
        """ Update the parameters with the given gradients
            Note: Use update method of an Array object to update it!

        Args:
            gradients (Dict[Array, np.ndarray]): Dictionary of parameter Array's and their gradients
        """
        for param in self.parameter_arrays:
            if param in gradients:
                grad = gradients[param]

                if self.l2_reg_coeff > 0:
                    grad += self.l2_reg_coeff * param.value
                
                param.update(param.value - self.lr * grad)