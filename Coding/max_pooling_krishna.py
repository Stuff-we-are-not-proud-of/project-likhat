import numpy as np

def max_pool(image_matrix, size=(2,2), stride=2):
    input_sizes = image_matrix.shape
    output_sizes = ((input_sizes[0] - size[0]) // stride + 1, (input_sizes[1] - size[1]) // stride + 1, input_sizes[2])
    output_matrix = np.zeros(shape=output_sizes)

    for i in range(output_sizes[2]):
        for j in range(output_sizes[0]):
            for k in range(output_sizes[1]):
                row_starter = i*stride
                column_starter = j*stride

                output_matrix[j,k,i] = np.max(image_matrix[row_starter:row_starter+size[0], column_starter:column_starter+size[1], i])
    
    return output_matrix