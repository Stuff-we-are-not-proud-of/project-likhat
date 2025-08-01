import numpy as np

def flatten(input_matrix):
    output_size = np.prod(input_matrix)
    flattened_output = np.reshape(input_matrix, (output_size, ))

    return flattened_output