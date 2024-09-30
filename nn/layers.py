from turtle import forward
from typing import Tuple, Callable, Any
from abc import ABC, abstractmethod
import numpy as np

from autograd.array import Array



class Module(ABC):
    """ Base class for every layer and network. """

    def __init__(self) -> None:
        self.parameters = []
        self.children_layers = []
        self.is_train_mode = True

    def train_mode(self):
        """ Change mode to training """
        self.is_train_mode = True
        for layer in self.children_layers:
            layer.train_mode()

    def eval_mode(self):
        """ Change mode to evaluation """
        self.is_train_mode = False
        for layer in self.children_layers:
            layer.eval_mode()

    def __call__(self, array: Array) -> Array:
        """ Forward calculation. Calls the forward function.

        Args:
            array (Array): Input Array of shape (B, D)

        Returns:
            Array: Output Array
        """
        return self.forward(array)

    @abstractmethod
    def forward(self, array: Array) -> Array:
        pass

    def __setattr__(self, __name: str, __value: Any) -> None:
        """ Save the parameters of the models that are assigned/set to a Module class.
        This method automatically keep tracks of the sub-models that are assigned to a Module class.

        Args:
            __name (str): Name of the attribute
            __value (Any): Value of the attribute

        """
        if isinstance(__value, Array) and __value.is_parameter:
            self.parameters.append(__value)
        if isinstance(__value, Module):
            self.parameters = [*self.parameters, *__value.parameters]
            self.children_layers.append(__value)
        return super().__setattr__(__name, __value)


class AffineLayer(Module):
    """ (Fully connected/Dense/Affine) layer.

    Args:
        in_size (int): Input feature size
        out_size (int): Output feature size
    """

    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        
        self.weight, self.bias = self.weight_init()

    def forward(self, array: Array) -> Array:
        """ Forward processing function. Only use Array operations so that you do not need to 
        worry about derivatives!!!

        Args:
            array (Array): Input Array of shape (B, D)

        Returns:
            Array: Output array: X @ W + b
        """
        
        val = array @ self.weight  + self.bias
        #print("VALUE SHAPE", val.shape)
        return  val
    

    def weight_init(self) -> Tuple[Array, Array]:
        """ Glorot Initialization for relu activation function and Normally distributed inputs.

        NOTE: The returned Arrays must be parameters!!!
              That is their ```is_parameter``` parameter must be set to True!!
              Otherwise, we can not use them in gradient calculations!

        Returns:
            Tuple[Array, Array]: Weight and bias Arrays
        """
        fan_in = self.in_size
        fan_out = self.out_size

        limit = np.sqrt(6 / (fan_in + fan_out))
        weight = Array(np.random.uniform(-limit, limit, size=(fan_in, fan_out)), is_parameter=True)
        #print("WEIGHT", weight.shape)
        bias = Array(np.zeros((1, fan_out)), is_parameter= True )
        #print("BIAS", bias.shape)
        
        
        return weight, bias


class ExampleFCN(Module):
    """ Example Fully Connected Network with a single affine layer

    Args:
        in_size (int): Input feature size
        out_size (int): Output feature size
    """

    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__() # This must be called at the top of every Layer and Network class

        self.in_size = in_size
        self.out_size = out_size

        self.layer_1 = AffineLayer(in_size, out_size)

    def forward(self, features: Array) -> Array:
        """ Forward processing function.

        Args:
            features (Array): Input of the network. 2D Array of shape (B, D)

        Returns:
            Array: Output of the network.
        """
        return self.layer_1(features) # Note that, this calls ```__call__```` method which calls ```forward``` method of layer_1


class FCN(Module):
    """ Basic Fully Connected Neural Network with at least two layers.

    Args:
        in_size (int): Input feature size
        out_size (int): Output feature size
        activation_fn (Callable[[Array], Array]): Activation Function (Non-linearity)
    """

    def __init__(self, in_size: int, out_size: int, activation_fn: Callable[[Array], Array]) -> None:
        super().__init__()
        
        self.in_size = in_size
        self.out_size = out_size
        self.activation_fn = activation_fn

        
        hidden_size = 64
        #print("OUTSIZE", out_size)
        self.layer_1 = AffineLayer(in_size, hidden_size)
        self.layer_2 = AffineLayer(hidden_size, out_size)

    def forward(self, features: Array) -> Array:
        """ Forward processing function.
        Note: When we use Array methods we do not need to worry about derivative.
              Only use Array methods!

        Args:
            features (Array): Input of the network. 2D Array of shape (B, D)

        Returns:
            Array: Output of the network.
            Note: Do not apply any non-linearity to the output of the function. We will use loss 
                function that expects logits (output of the affine layer).
        """
       
        hidden_output = self.activation_fn(self.layer_1(features))
        output = self.layer_2(hidden_output)
        return output

class BatchNorm(Module):
    """ Batch Norm Layer. Use running mean and running std in evaluation mode. Update the
    running statistics at every forward processing.

    Args:
        feature_size (int): Input feature size
    """

    def __init__(self, feature_size: int) -> None:
        super().__init__()
        
        self.feature_size = feature_size
        self.is_training_mode = True

    def forward(self, array: Array, epsilon: float = 1e-7) -> Array:
        """ Forward processing function.
        Note: When we use Array methods we do not need to worry about derivative.
              Only use Array methods!
        Note: The behavior of this function changes based on the ```is_training_mode``` attribute

        Args:
            array (Array): Input Array
            epsilon (float, optional): Small value for divisions. Defaults to 1e-7.

        Returns:
            Array: Output of the layer
        """
        
        
        running_mean = Array(np.zeros(self.feature_size))
        running_std = Array(np.ones(self.feature_size))
        
        gamma = Array(np.ones(self.feature_size))
        beta = Array(np.zeros(self.feature_size))  
        momentum = Array(np.array([0.1] * self.feature_size ))
        epsilon = Array(np.array([epsilon] *self.feature_size))
        
        if self.is_training_mode:
            batch_mean = array.mean(axis=0)
            batch_std = self.std(array, axis=0)
            
     
            running_mean = momentum * batch_mean + (1 - momentum) * running_mean
            running_std = momentum * batch_std + (1 - momentum) * running_std
            
            mean = batch_mean
            std = batch_std
             
        else:
            mean = running_mean
            std = running_std
            
        #print("ARR", array.shape)
        #print("MEAN", mean.shape)
        #print("STD", std)
        #print("EPS", epsilon)
        normalized_array = (array - mean) / (std + epsilon)
        output_array =   gamma *normalized_array + beta
        return output_array

    @staticmethod
    def std(array: Array, axis: int, keepdims: bool = True) -> Array:
        """ Calculate Standard Deviation of an Array on an axis. 
        You can use this function in forward!

        Args:
            array (Array): Input Array of shape (B, D)
            axis (int): Axis of operation
            keepdims (bool, optional): If true axis of operation will not be removed but it will
                be reduced to 1. Defaults to True.

        Returns:
            Array: Output Array of shape (B, 1) if axis=1 and keepdims=True
        """
        mean = array.mean(axis=axis, keepdims=True)
        array_bar = (array - mean)
        return (array_bar * array_bar).mean(axis=axis, keepdims=keepdims)


class BatchNormFCN(Module):
    """ Fully Connected Network with Batch Norm Layers

    Args:
        in_size (int): Input feature size
        out_size (int): Output size
        activation_fn (Callable[[Array], Array]): Activation Function (Non-linearity)
    """

    def __init__(self, in_size: int, out_size: int, activation_fn: Callable[[Array], Array]) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.activation_fn = activation_fn
        
        self.affine = AffineLayer(in_size, out_size)
        self.batch_norm = BatchNorm(out_size)

    def forward(self, features: Array) -> Array:
        """ Forward processing function

        Args:
            features (Array): Input Array of shape (B, D)

        Returns:
            Array: Output of the network
            Note: Do not apply any non-linearity to the output of the function. We will use loss 
                function that expects logits (output of the affine layer).
        """

        affine_output = self.affine(features)
        normalized_output = self.batch_norm.forward(affine_output)
        activated_output = self.activation_fn(normalized_output)
        
        return activated_output



class Dropout(Module):

    def __init__(self, drop_p: float) -> None:
        """ Dropout Layer. In traning mode, randomly drop some features. In evaluation mode, do nothing!

        Args:
            drop_p (float): Probability of dropping.

        Raises:
            ValueError: Drop probability must be in the range [0, 1]
        """
        super().__init__()
        if not 0 <= drop_p <= 1:
            raise ValueError("Drop probability must be in the range [0, 1]")
        self.drop_p = drop_p

    def forward(self, array: Array) -> Array:
        """ Forward processing function. Randomly drop input Array if in training mode.

        Args:
            array (Array): Input Array of shape (B, D)

        Returns:
            Array: Output Array of same shape
        """
        if self.is_train_mode:
            mask = Array((np.random.rand(*array.shape) > self.drop_p).astype(array.dtype))
            array =  array * mask / (1 - self.drop_p)
        
        return array



class DropoutFCN(Module):
    """ Fully connected network with Batch Norm and Dropout layers.

    Args:
        in_size (int): Input feature size
        out_size (int): Output size
        activation_fn (Callable[[Array], Array]): Activation Function (Non-linearity)
        p_drop (float, optional): Dropping probability for dropout layer(s). Defaults to 0.1.
    """

    def __init__(self, in_size: int, out_size: int, activation_fn: Callable[[Array], Array], p_drop: float = 0.1) -> None:
        super().__init__()

        self.fc = AffineLayer(in_size, out_size)
        self.batch_norm = BatchNorm(out_size)
        self.dropout = Dropout(p_drop)
        self.activation_fn = activation_fn
        
    def forward(self, features: Array) -> Array:
        """ Forward processing function.

        Args:
            features (Array): Input Array of shape (B, D)

        Returns:
            Array: Output Array.
            Note: Do not apply any non-linearity to the output of the function. We will use loss 
                function that expects logits (output of the affine layer).
        """
        x = self.fc(features)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.activation_fn(x)
        return x


class MaxoutFCN(Module):

    def __init__(self, in_size: int, out_size: int, p_drop: float = 0.1, n_affine_outputs: int = 5) -> None:
        """ Fully connected network with Maxout, Batch Norm and Dropout layers. We do not need 
        to use an activation function here!

        Args:
            in_size (int): Input feature size
            out_size (int): Output size
            p_drop (float, optional): Dropping probability for dropout layer(s). Defaults to 0.1.
            n_affine_outputs (int, optional): Number of affine outputs to be max out for every 
                feature. Defaults to 5.
        """        
        super().__init__()
        
        self.in_size = in_size
        self.out_size = out_size
        self.feature_size = 128
        self.n_affine_outputs = n_affine_outputs
        
        
        self.fc1 = AffineLayer(in_size, 64 * n_affine_outputs)
        self.batch_norm = BatchNorm(in_size)
        self.dropout = Dropout(p_drop)
        self.fc2 = AffineLayer(64, out_size)


    def forward(self, features: Array) -> Array:
        """ Forward processing function

        Args:
            features (Array): Input Array of shape (B, D)

        Returns:
            Array: Output Array.
            Note: Do not apply any non-linearity to the output of the function. We will use loss 
                function that expects logits (output of the affine layer).
        """
        
        x = self.batch_norm(features)
        print("X", x.shape)
        x = self.fc1(features)
        print("X", x.shape)
        x = x.reshape(features.shape[0],-1,64)
        
    
        
        print("X", x.shape)
        #print("IN", self.in_size)
        #print("OUT", self.out_size)

        x = self.dropout(x)
        
        #print("X", x.shape)
        x = x.max(axis = 1)
        
        print("X", x.shape)
        x = self.fc2(x)
        
        return x



