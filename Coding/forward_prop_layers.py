import numpy as np

def dropout(input_array, dropout_rate=0.5, training=True):
    if not training:
        return input_array, {}
    keep_prob = 1 - dropout_rate
    mask = np.random.binomial(1, keep_prob, size=input_array.shape) / keep_prob 
    
    masked_output = input_array * mask
    cache = {
        'mask': mask
    }
    return masked_output, cache

def flatten(input_matrix):
    output_size = int(np.prod(input_matrix.shape))
    flattened_output = np.reshape(input_matrix, (output_size, ))
    cache = {
        'input_shape': input_matrix.shape
    }
    return flattened_output, cache

import numpy as np

def densen(input_array, output_size, activation="ReLU", W=None, b=None):
    input_size = len(input_array)
    if W is None or b is None:
        limit = np.sqrt(6 / input_size)
        W = np.random.uniform(-limit, limit, size=(input_size, output_size))
        b = np.zeros(output_size)

    multiplied_matrix = input_array @ W
    z = multiplied_matrix + b
    output = z.copy()
    if activation == "ReLU":
        output = np.maximum(0, output)
    elif activation == "sigmoid":
        z = np.clip(z, -500, 500)
        output = 1 / (1 + np.exp(-z))
    
    cache = {
        'Z': z,
        'A': output,
        'weights': W,
        'biases': b,
        'input': input_array
    }
    return output, cache