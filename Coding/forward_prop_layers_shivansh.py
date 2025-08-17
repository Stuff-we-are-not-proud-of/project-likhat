import numpy as np

def densen(input_array, output_size, activation="ReLU"):
    input_size = len(input_array)
    fan_in = input_size
    fan_out = output_size
    limit = np.sqrt(6 / (fan_in + fan_out))
    
    weights = np.random.uniform(-limit, limit, size=(input_size, output_size))
    biases = np.zeros(output_size)

    z = input_array @ weights + biases

    if activation == "ReLU":
        output = np.maximum(0, z)
    elif activation == "sigmoid":
        output = 1 / (1 + np.exp(-z))
    else:
        output = z

    cache = {
        "input": input_array,
        "weights": weights,
        "biases": biases,
        "z": z,
        "activation": activation
    }
    return output, cache


def densen(input_array, output_size, activation="ReLU"):
    input_size = len(input_array)
    fan_in = input_size
    fan_out = output_size
    limit = np.sqrt(6 / (fan_in + fan_out))
    
    weights = np.random.uniform(-limit, limit, size=(input_size, output_size))
    biases = np.zeros(output_size)

    z = input_array @ weights + biases

    if activation == "ReLU":
        output = np.maximum(0, z)
    elif activation == "sigmoid":
        output = 1 / (1 + np.exp(-z))
    else:
        output = z

    cache = {
        "input": input_array,
        "weights": weights,
        "biases": biases,
        "z": z,
        "activation": activation
    }
    return output, cache

def flatten(input_matrix):
    original_shape = input_matrix.shape
    flattened_output = np.reshape(input_matrix, (-1,))
    cache = {"original_shape": original_shape}
    return flattened_output, cache
