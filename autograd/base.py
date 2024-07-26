from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np


class BaseArray(ABC):
    """ Base Array Class """
    pass


class BaseOperation(ABC):
    """ Base Opeartion Class """

    def __init__(self, *operands: List[BaseArray]) -> None:
        self.operands = operands

    @abstractmethod
    def __call__(self) -> np.ndarray:
        """ Calculate and return the NumPy array of the corresponding operation

        Returns:
            np.ndarray: Resultant NumPy array
        """

    @abstractmethod
    def vjp(self, grad_vector: np.ndarray) -> Tuple[np.ndarray]:
        """ Calculate Vector Jacobian Product (vjp) using the gradient vector and the inputs of 
        the operation.

        Args:
            grad_vector (np.ndarray): Gradient vector coming from subsequent operation in the 
                ordered computational graph.

        Returns:
            Tuple[np.ndarray]: Results of vjp operation w.r.t all inputs
        """
        pass
