import numpy as np

def max_pool_backward(dout, cache):

    input_matrix = cache["input"]
    size = cache["size"]
    stride = cache["stride"]
    H_out, W_out, C = cache["pooled_shape"]

    dx = np.zeros_like(input_matrix)

    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                region = input_matrix[h_start:h_start+size, w_start:w_start+size, c]
                max_val = np.max(region)
                mask = (region == max_val)
                dx[h_start:h_start+size, w_start:w_start+size, c] += mask * dout[i, j, c]

    return dx
