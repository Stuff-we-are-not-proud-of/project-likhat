import numpy as np

def flatten(input_matrix):
    output_size = int(np.prod(input_matrix.shape))
    flattened_output = np.reshape(input_matrix, (output_size, ))

    return flattened_output