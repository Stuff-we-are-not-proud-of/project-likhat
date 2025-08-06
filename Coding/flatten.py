import numpy as np
def flatten(input_matrix):
    output_size = np.prod(input_matrix.shape)
    flattened_output = np.reshape(input_matrix, (int(output_size), ))
    return flattened_output
