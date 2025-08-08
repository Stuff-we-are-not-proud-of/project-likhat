import numpy as np

def maxpool_backward(dA, cache):
    input = cache['input']
    size = cache['size']
    stride = cache['stride']
    input_height, input_width, input_channels = input.shape
    dX = np.zeros_like(input)
    
    output_height, output_width, _ = dA.shape
    
    for i in range(output_height):
        for j in range(output_width):
            for k in range(input_channels):
                row_start = i * stride
                col_start = j * stride
                sub = input[row_start:row_start + size[0], col_start:col_start + size[1], k]
                max_mask = sub == np.max(sub)
                dX[row_start:row_start + size[0], col_start:col_start + size[1], k] += max_mask * dA[i, j, k]
    
    return dX