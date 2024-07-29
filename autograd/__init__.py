from typing import Callable, List, Dict, Optional
from collections import defaultdict
import numpy as np

from autograd.array import Array


def topological_sort(root_array: Array) -> List[Array]:
    """ Topologically sort the graph given by the root Array. Topological sorting is 
    performend in the "reversed graph".

    Args:
        root_array (Array): Root/starting Array in the reversed graph. (One 
            that we want to differentiate)

    Returns:
        List[Array]: Ordered arrays in a list. First element must be the root Array.
    """   
    ordered_arrays = []

    def traverse(array: Array):
        if array not in ordered_arrays:
            if array.operation is not None:
                for arr in array.operation.operands:
                    traverse(arr)
            ordered_arrays.append(array)

    traverse(root_array)
    return ordered_arrays[::-1]



def grad(variable: Array, gradient_vector: Optional[np.ndarray] = None) -> Dict[Array, np.ndarray]:
    """ Reverse-mode differentiation. Calculate the gradient of the given variable using vjp 
    functions of each array in the topologically sorted computational graph.

    Args:
        variable (Array): Operand Array
        gradient_vector (Optional[np.ndarray], optional): Initial gradient vector of the 
            first vjp operation. Defaults to scalar 1.0.

    Raises:
        ValueError: If gradient_vector is not given and the variable is not scalar

    Returns:
        Dict[Array, np.ndarray]: Dictionary of Array and their corresponding gradients
        Only include Array's in the dictionary if they are parameter (Array.is_parameter==True)
    """
    
    sorted_arrays = topological_sort(variable)
    


    gradients = defaultdict(list)
    

    gradients[variable] = [np.float32(1.0) if gradient_vector is None else gradient_vector]
    
    #print("GRADIENTS", gradients)
    
    for var in sorted_arrays:
        #print("VARSSS", var)
        
        
        in_grad = sum(gradients[var])
        
        #print("in grad", in_grad)
        #print("VAR OPERATION", var.operation)

        if var.operation is not None:
            
            #print("VAR OPERATION", var.operation)
            
            out_grads = var.operation.vjp(in_grad)
            
            #print("out grad", out_grads)
            
            if not isinstance(out_grads, tuple):
                out_grads = (out_grads,) 
            for child, out in zip(var.operation.operands, out_grads):
                gradients[child].append(out)
        
            
    #print(var.operation.operands, "operands")
    #print(gradients, "grads")            
    #print("result", {var: sum(gradients[var]) for var in sorted_arrays if var.is_parameter})
    
    
    #gradients = {array: grad for array, grad in gradients.items() if array.is_parameter}
    return {var: sum(gradients[var]) for var in sorted_arrays if var.is_parameter}



