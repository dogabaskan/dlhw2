from autograd.array import Array
import numpy as np



def relu(array: Array) -> Array:
    """ Rectified Linear Unit activation function

    Args:
        array (Array): Input Array

    Returns:
        Array: Nonlinearity applied Array
    """
    return array.maximum(0.0)


def sigmoid(array: Array) -> Array:
    """ Sigmoid activation function

    Args:
        array (Array): Input Array

    Returns:
        Array: Nonlinearity applied Array
    """
    return array.sigmoid()


def tanh(array: Array) -> Array:
    """ Tanh activation function

    Args:
        array (Array): Input Array

    Returns:
        Array: Nonlinearity applied Array
    """
    return array.tanh()


def leaky_relu(array: Array, negative_slope: float = 0.01) -> Array:
    """ Leaky Rectified Linear Unit activation function
        Note: Only use Array methods!

    Args:
        array (Array): Input Array
        negative_slope (float): Slope of the negative region

    Returns:
        Array: Nonlinearity applied Array
    """

    
    return (array.maximum(0) + array.minimum(0)* negative_slope)


def softmax(array: Array) -> Array:
    """ Softmax function

    Args:
        array (Array): Input Array of shape (B, D)

    Returns:
        Array: Nonlinearity applied Array
    """
    exp = array.exp()
    return exp / exp.sum(-1, keepdims=True)


def nll_with_logits_loss(array: Array, label: Array) -> Array:
    """ Negative Log Likelihood that expects logits instead of class probabilities.
        Note: Only use Array methods!
    Args:
        array (Array): Logit Array of shape (B, D)
        label (Array): Label Array of shape (B)

    Returns:
        Array: Nonlinearity applied Array
    """
    exp_array = array.exp()
    sum_exp_array = exp_array.sum(axis=1, keepdims=True)
    probs = exp_array / sum_exp_array

    log_probs = probs.log()
    
    #print("ARR SHAPE" , array.shape)
    
    one_hot_labels = Array.onehot(label, array.shape[-1])
    
    selected_log_probs = (log_probs * one_hot_labels).sum(axis=1)

    

    return -selected_log_probs
