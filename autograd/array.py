from multiprocessing import ProcessError
from typing import List, Tuple, Optional, Union, Any
from itertools import chain
from uuid import uuid4
import numpy as np

from autograd.base import BaseArray, BaseOperation
from autograd.operations import Add, Subtract, Multiply, Divide, Maximum, Minimum, Sum, Mean, Max, Matmul, Reshape, Exp, Log, Power, Tanh, Sigmoid, Onehot


class Array(BaseArray):
    """ Fundamental data class of autograd package. Array holds data as a NumPy array and provides
    basic mathematical operations (similar to NumPy) such as sum, reshape, matrix multiplication, 
    ...etc. On top of that, Array objects trace their creation history and they are immutable
    (to simplify tracing).

    Args:
        value (Union[np.ndarray, np.float32]): NumPy array as the data of the array
        operation (BaseOperation, optional): Operation that results to creation of this 
            Array. Defaults to None if there is no such operation.
        is_parameter (bool, optional): If true, this array is considered as parameter.
            Defaults to False.
    """

    def __init__(self, value: Union[np.ndarray, np.float32],
                 operation: BaseOperation = None,
                 is_parameter: bool = False) -> None:
        self.is_parameter = is_parameter
        self.value = value
        if isinstance(self.value, np.ndarray):
            self.value.flags.writeable = False

        self.operation = operation
        self.hash_code = hash(str(uuid4()))

    def __repr__(self) -> str:
        op_str = "" if self.operation is None else f", Operation: {self.operation.__class__.__name__}"
        return f"Autograd array{op_str}: {self.value}"

    def __hash__(self) -> str:
        """ Hash code of the Array object

        Returns:
            str: Hash code
        """
        return self.hash_code

    def __setattr__(self, name: str, value: Any) -> None:
        """ Attribute setter. Make Array objects immutable.

        Args:
            name (str): Name of the attribute to be assigned
            value (Any): Value of the attribute to be assigned

        Raises:
            AttributeError: If attribute name exists.
        """
        if name in self.__dict__.keys():
            raise AttributeError(f"Attribute {name} can not be reassigned!")
        super().__setattr__(name, value)

    def update(self, value: np.ndarray) -> None:
        """ Update the value of Array.

        Args:
            value (np.ndarray): New values
        """
        self.value.flags.writeable = True
        self.value[:] = value[:]
        self.value.flags.writeable = False

    @staticmethod
    def to_array(value: Union[float, int, np.float32, "Array"]) -> "Array":
        """ Convert NumPy arrays to Array.

        Args:
            value (Union[float, np.float32, "Array"]): NumPy array or Array

        Returns:
            Array: Array object
        """
        if isinstance(value, (float, int)):
            return Array(value=np.array(value, dtype=np.float32))
        return value

    @property
    def shape(self) -> Tuple[int]: return self.value.shape

    @property
    def dtype(self) -> Any: return self.value.dtype

    @property
    def size(self) -> int: return self.value.size

    @property
    def ndim(self) -> int: return self.value.ndim

    def __add__(self, other: "Array") -> "Array":
        op = Add(self, self.to_array(other))
        return Array(op(), operation=op)

    def __radd__(self, other: "Array") -> "Array":
        op = Add(self.to_array(other), self)
        return Array(op(), operation=op)

    def __mul__(self, other: "Array") -> "Array":
        op = Multiply(self, self.to_array(other))
        return Array(op(), operation=op)

    def __rmul__(self, other: "Array") -> "Array":
        op = Multiply(self.to_array(other), self)
        return Array(op(), operation=op)

    def __truediv__(self, other: "Array") -> "Array":
        op = Divide(self, self.to_array(other))
        return Array(op(), operation=op)

    def __rtruediv__(self, other: "Array") -> "Array":
        op = Divide(self.to_array(other), self)
        return Array(op(), operation=op)

    def __sub__(self, other: "Array") -> "Array":
        op = Subtract(self, self.to_array(other))
        return Array(op(), operation=op)

    def __rsub__(self, other: "Array") -> "Array":
        op = Subtract(self.to_array(other), self)
        return Array(op(), operation=op)

    def __neg__(self) -> "Array":
        op = Multiply(self, self.to_array(-1.0))
        return Array(op(), operation=op)

    def __pow__(self, other: int) -> "Array":
        op = Power(self, other)
        return Array(op(), operation=op)

    def __matmul__(self, other) -> "Array":
        op = Matmul(self, other)
        return Array(op(), operation=op)

    def maximum(self, other: "Array") -> "Array":
        op = Maximum(self, self.to_array(other))
        return Array(op(), operation=op)

    def minimum(self, other: "Array") -> "Array":
        op = Minimum(self, self.to_array(other))
        return Array(op(), operation=op)

    def reshape(self, *shape: int) -> "Array":
        op = Reshape(self, *shape)
        return Array(op(), operation=op)

    def sum(self, axis: int = 0, keepdims: bool = False) -> "Array":
        op = Sum(self, axis, keepdims)
        return Array(op(), operation=op)

    def mean(self, axis: int = 0, keepdims: bool = False) -> "Array":
        
        op = Mean(self, axis, keepdims)
        return Array(op(), operation=op)

    def max(self, axis: int = 0, keepdims: bool = False) -> "Array":
        op = Max(self, axis, keepdims)
        return Array(op(), operation=op)

    def exp(self) -> "Array":
        op = Exp(self)
        return Array(op(), operation=op)

    def log(self) -> "Array":
        op = Log(self)
        return Array(op(), operation=op)

    def sigmoid(self) -> "Array":
        op = Sigmoid(self)
        return Array(op(), operation=op)

    def tanh(self) -> "Array":
        op = Tanh(self)
        return Array(op(), operation=op)

    def onehot(self, max_size: int) -> "Array":
        op = Onehot(self, max_size)
        return Array(op(), operation=op)
