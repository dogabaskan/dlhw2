from typing import Tuple
import numpy as np

from autograd.base import BaseArray, BaseOperation


class SimpleAdd(BaseOperation):

    def __init__(self, first_operand: BaseArray, second_operand: BaseArray) -> None:
        """ Add operation that do not support Auto-Broadcastion (at derivative calculations).
        This is an example operation!

        Args:
            first_operand (BaseArray): First Array
            second_operand (BaseArray): Second Array

        Raises:
            ValueError: If shape of the first and the second arrays are not the same
        """
        if first_operand.shape != second_operand.shape:
            raise ValueError("Shape mismatch")
        self.first_operand = first_operand
        self.second_operand = second_operand
        super().__init__(first_operand, second_operand)

    def __call__(self) -> np.ndarray:
        """ Calculate and return the NumPy array of the summation

        Returns:
            np.ndarray: Resultant NumPy array
        """
        return self.first_operand.value + self.second_operand.value

    def vjp(self, grad_vector: np.ndarray) -> Tuple[np.ndarray]:
        """ Calculate Vector Jacobian Product (vjp) using the gradient vector and the inputs of 
        the operation.

        Args:
            grad_vector (np.ndarray): Gradient vector coming from subsequent operation in the 
                ordered computational graph.

        Raises:
            ValueError: If the shape of gradient vector does not match with the inputs shape

        Returns:
            Tuple[np.ndarray]: Results of vjp operation w.r.t both inputs
        """
        if grad_vector.shape != self.first_operand.shape:
            raise ValueError("Gradient vector shape mismatch")
        return grad_vector, grad_vector


class Sigmoid(BaseOperation):
    """ Sigmoid Operation """

    def __call__(self) -> np.ndarray:
        value = np.clip(self.operands[0].value, -50, 50) # Precaution for numeric issues
        return 1/(1 + np.exp(-value))

    def vjp(self, grad_vector: np.ndarray) -> np.ndarray:
        sigmoid = self()
        return (sigmoid * (1 - sigmoid)) * grad_vector


class Tanh(BaseOperation):
    """ Tanh Operation """

    def __call__(self) -> np.ndarray:

        return np.tanh(self.operands[0].value)
    

    def vjp(self, grad_vector: np.ndarray) -> np.ndarray:
        
        tanh_val = self()
        grad =  (1- tanh_val**2) * grad_vector
        
        
        #print("1" , self.operands[0].value)
        #print("2" , tanh_val)
        #print("3", grad_vector)
        #print("grad", grad , "grad")

        return grad

class Onehot(BaseOperation):
    """ Onehot Operation

    Args:
        operand (BaseArray): 1D int Array object
        max_size (int): Length of each onehot vector

    Raises:
        ValueError: If input Array is not 1D
    """

    def __init__(self, operand: BaseArray, max_size: int) -> None:
        super().__init__(operand)
        if operand.ndim != 1:
            raise ValueError("Input must be 1 dimensional")
        self.max_size = max_size

        
    def __call__(self) -> np.ndarray:
        
       
        
        onehot_encoded = np.eye(self.max_size)[self.operands[0].value]
        return onehot_encoded


    def vjp(self, grad_vector: np.ndarray) -> np.ndarray:
        grad_input = np.zeros(self.operands[0].shape, dtype=grad_vector.dtype)
        for i, index in enumerate(self.operands[0].value):
            grad_input[i] = grad_vector[i, index]
        return grad_input



class Power(BaseOperation):
    """ Power Operation

    Args:
        operand (BaseArray): Input Array
        pow (int): Power
    """

    def __init__(self, operand: BaseArray, pow: int) -> None:
        super().__init__(operand)
        self.pow = pow

    def __call__(self) -> np.ndarray:
        return np.power(self.operands[0].value, self.pow)
    
    
    def vjp(self, grad_vector: np.ndarray) -> np.ndarray:
        
        grad = grad_vector * (self.pow) * np.power(self.operands[0].value, (self.pow - 1))
        
        return grad


class Exp(BaseOperation):
    """ Exponential Operation """

    def __call__(self) -> np.ndarray:
        return np.exp(self.operands[0].value)

    def vjp(self, grad_vector: np.ndarray) -> np.ndarray:
        
        grad = grad_vector * np.exp(self.operands[0].value)
        return grad


class Log(BaseOperation):
    """ Logarithm Operation """

    def __call__(self) -> np.ndarray:
        return np.log(self.operands[0].value)

    def vjp(self, grad_vector: np.ndarray) -> np.ndarray:

        grad = grad_vector / self.operands[0].value 
        return grad

class BroadcastedOperation(BaseOperation):
    """ Base class for Broadcastable operations of two inputs.

    Args:
        first_operand (BaseArray): First array
        second_operand (BaseArray): Second 
    """

    def __init__(self, first_operand: BaseArray, second_operand: BaseArray) -> None:
        self.first_operand = first_operand
        self.second_operand = second_operand
        super().__init__(first_operand, second_operand)

    @staticmethod
    def broadcast_vjp(grad_vector: np.ndarray, original_shape: Tuple[int]) -> np.ndarray:
        """ Derivative of broadcasting. vjp for broadcasting. This function is used for 
        broadcastable operations. ---> REDUCING

        Args:
            grad_vector (np.ndarray): Gradient vector as NumPy array given from the broadcasted 
                operation
            original_shape (Tuple[int]): Shape of the operand before broadcasting

        Raises:
            ValueError: If original_shape and the shape of grad_vector are not appropriate for 
                broadcasting

        Returns:
            np.ndarray: NumPy array of derivative of broadcasting
        """
        print("GRAD", grad_vector.shape)        
        num_dim_grad = len(grad_vector.shape)
        num_dim_origin = len(original_shape)
        difference = num_dim_grad - num_dim_origin      
        
        my_list =  []
                
        originals = list(original_shape)
        originals = originals[::-1]
        
               
  
        for i in range(difference):
            originals.append(1)
            
        
        originals = originals[::-1]
        
        i = 0                   
        
        while i < num_dim_grad:
            '''CHECK HERE'''
            if original_shape == 0:
                my_list = range(len(grad_vector.shape))
            if grad_vector.shape[i] != originals[i]: 
                my_list.append(i)
                if grad_vector.shape[i] != 1:
                    if originals[i] != 1:
                        raise ValueError
            i += 1
        
        
        axises = tuple(my_list)
        print("OG SHAPE", np.sum(grad_vector, axis = axises))
        derivative = np.sum(grad_vector, axis = axises).reshape(original_shape)
        return derivative
        

class Add(BroadcastedOperation):
    """ Broadcastable Addition operation """

    def __call__(self) -> np.ndarray:
        
        #print('FIRST', self.first_operand)
        #print("SECOND", self.second_operand)
        
        return np.add([self.first_operand.value], [self.second_operand.value])

    def vjp(self, grad_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return BroadcastedOperation.broadcast_vjp(grad_vector, self.first_operand.shape),\
            BroadcastedOperation.broadcast_vjp(grad_vector, self.second_operand.shape)
            

class Subtract(BroadcastedOperation):
    """ Broadcastable Subtraction operation """

    def __call__(self) -> np.ndarray:
        return np.subtract(self.first_operand.value, self.second_operand.value)

    def vjp(self, grad_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return BroadcastedOperation.broadcast_vjp(grad_vector, self.first_operand.shape), \
           - BroadcastedOperation.broadcast_vjp(grad_vector, self.second_operand.shape)



class Multiply(BroadcastedOperation):
    """ Broadcastable Multiplication operation """

    def __call__(self) -> np.ndarray:
        return np.multiply(self.first_operand.value, self.second_operand.value)
    
    
    def vjp(self, grad_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        # print("GRAAAAAD", grad_vector.shape)
        first_grad = grad_vector* self.second_operand.value
        second_grad = grad_vector * self.first_operand.value
        
        return BroadcastedOperation.broadcast_vjp(first_grad, self.first_operand.shape),\
            BroadcastedOperation.broadcast_vjp(second_grad,self.second_operand.shape)    



class Divide(BroadcastedOperation):
    """ Broadcastable Division operation """

    def __call__(self) -> np.ndarray:
        return np.divide(self.first_operand.value, self.second_operand.value)

    def vjp(self, grad_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        first_grad = grad_vector / self.second_operand.value
        
        second_grad = - grad_vector * self.first_operand.value / (self.second_operand.value ** 2)
        
        return BroadcastedOperation.broadcast_vjp(first_grad, self.first_operand.value.shape), \
               BroadcastedOperation.broadcast_vjp(second_grad, self.second_operand.value.shape)
               
               
class Maximum(BroadcastedOperation):
    """ Broadcastable Maximum operation """

    def __call__(self) -> np.ndarray:
        return np.maximum(self.first_operand.value, self.second_operand.value)

    def vjp(self, grad_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        first_weight = self.first_operand.value >= self. second_operand.value
        second_weight = self.second_operand.value > self.first_operand.value
        
        first_grad = grad_vector * first_weight
        second_grad = grad_vector * second_weight
        
        return BroadcastedOperation.broadcast_vjp(first_grad, self.first_operand.value.shape), \
               BroadcastedOperation.broadcast_vjp(second_grad, self.second_operand.value.shape)
               
               
class Minimum(BroadcastedOperation):
    """ Broadcastable Minimum operation """

    def __call__(self) -> np.ndarray:
        return np.minimum(self.first_operand.value, self.second_operand.value)

    def vjp(self, grad_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        first_weight = self.first_operand.value <= self.second_operand.value
        second_weight = self.second_operand.value < self.first_operand.value
        
        first_grad = grad_vector * first_weight
        second_grad = grad_vector * second_weight
        
        return BroadcastedOperation.broadcast_vjp(first_grad, self.first_operand.value.shape), \
            BroadcastedOperation.broadcast_vjp(second_grad, self.second_operand.value.shape)
            
            
            
            

class ReduceOperations(BaseOperation):
    """ Base class for reduce operation. This operation reduces the dimension that it operates

    Args:
        operand (BaseArray): Input Array
        axis (int, optional): Operation axis. Defaults to 0.
        keepdims (bool, optional): If true keep the dimension of operation. Defaults to False.
    """

    def __init__(self, operand: BaseArray, axis: int = 0, keepdims: bool = False) -> None:
        self.operand = operand
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(operand)

    @staticmethod
    def reduce_vjp(grad_vector: np.ndarray, axis: int, keepdims: bool, original_shape: Tuple[int]) -> np.ndarray:
        """ Derivative of reduction operation. (vjp)

        Args:
            grad_vector (np.ndarray): Gradient vector as NumPy array given from the reduced 
                operation.
            axis (int): Axis of operation.
            keepdims (bool): If true keep the dimension of operation.
            original_shape (Tuple[int]): Original shape of the operand Array before reduction.

        Returns:
            np.ndarray: NumPy array of the derivative of reduce
        """

        if not keepdims:
            if not isinstance(axis, tuple):
                axis = (axis,)
            for ax in sorted(axis):
                grad_vector = np.expand_dims(grad_vector, ax)
        
        # Broadcast the gradient to the original shape
        grad_vector = np.broadcast_to(grad_vector, original_shape)
        
        return grad_vector

        


class Sum(ReduceOperations):
    """ Reducing Summation operation """

    def __call__(self) -> np.ndarray:
        return np.sum(self.operand.value, axis = self.axis, keepdims = self.keepdims)

    def vjp(self, grad_vector: np.ndarray) -> np.ndarray:
        return ReduceOperations.reduce_vjp(grad_vector,self.axis , self.keepdims, self.operand.shape)


class Mean(ReduceOperations):
    """ Reducing Mean operation """

    def __call__(self) -> np.ndarray:
        return np.mean(self.operand.value, axis = self.axis , keepdims = self.keepdims)

    def vjp(self, grad_vector: np.ndarray) -> np.ndarray:
        
        original_shape = self.operand.value.shape
        
        if self.axis is None:
            num_elements = np.prod(original_shape)
        else:
            if isinstance(self.axis, int):
                self.axis = (self.axis,)
            num_elements = np.prod([original_shape[ax] for ax in self.axis])

        scaled_grad = grad_vector / num_elements

        return ReduceOperations.reduce_vjp(scaled_grad, self.axis, self.keepdims, original_shape)


class Max(ReduceOperations):
    """ Reducing Max operation """

    def __call__(self) -> np.ndarray:
        
                
        return np.max(self.operand.value , axis = self.axis, keepdims= self.keepdims)

    def vjp(self, grad_vector: np.ndarray) -> np.ndarray:
          
        max_val = np.max(self.operand.value, axis=self.axis, keepdims=True)
        
        mask = self.operand.value == max_val
        
        if not self.keepdims:
            grad_vector = np.expand_dims(grad_vector, axis=self.axis)
        
        grad_output = mask * grad_vector
        
        grad_output = self.reduce_vjp(grad_output, axis=self.axis, keepdims=True, original_shape=self.operand.shape)
        
        return grad_output

class ShapeOperations(BaseOperation):
    """ Base Shape operation """

    def __init__(self, operand: BaseArray) -> None:
        super().__init__(operand)
        self.operand = operand
        self.original_shape = operand.shape


class Reshape(ShapeOperations):
    """ Reshape operation

    Args:
        operand (BaseArray): Input Array to be reshaped
        shape (*int): New shape of the Array
    """

    def __init__(self, operand: BaseArray, *shape: int) -> None:
        super().__init__(operand)
        self.shape = shape

    def __call__(self) -> np.ndarray:
        return self.operand.value.reshape(*self.shape)

    def vjp(self, grad_vector: np.ndarray) -> np.ndarray:
        return grad_vector.reshape(*self.original_shape)


class Matmul(BaseOperation):
    """ Matrix Multiplication Operation

    Args:
        first_operand (BaseArray): Left Array
        second_operand (BaseArray): Right Array
    """

    def __init__(self, first_operand: BaseArray, second_operand: BaseArray) -> None:
        self.first_operand = first_operand
        self.second_operand = second_operand
        super().__init__(first_operand, second_operand)

    def __call__(self) -> np.ndarray:
        return np.dot(self.first_operand.value, self.second_operand.value)
    
    def vjp(self, grad_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        grad_first_operand = np.dot(grad_vector, self.second_operand.value.T)
        grad_second_operand = np.dot(self.first_operand.value.T, grad_vector)
        return grad_first_operand, grad_second_operand