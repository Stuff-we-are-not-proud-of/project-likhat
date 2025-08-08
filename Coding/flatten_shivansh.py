import numpy as np

def flatten(input_matrix):
    original_shape = input_matrix.shape
    flattened_output = np.reshape(input_matrix, (-1,))
    cache = {"original_shape": original_shape}
    return flattened_output, cache
