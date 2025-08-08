import numpy as np

def conv_backward(dout, cache):

    input_matrix = cache["input"]
    filters = cache["filters"]
    biases = cache["biases"]
    stride = cache["stride"]
    padding = cache["padding"]
    H_out, W_out, num_filters = cache["output_shape"]

    H_in, W_in, C_in = input_matrix.shape
    fN, fH, fW, _ = filters.shape

    dx = np.zeros_like(input_matrix)
    dfilters = np.zeros_like(filters)
    dbiases = np.zeros_like(biases)

    # Pad input and dx
    pad_input = np.pad(input_matrix, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    pad_dx = np.pad(dx, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

    for k in range(num_filters):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                region = pad_input[h_start:h_start+fH, w_start:w_start+fW, :]
                dfilters[k] += region * dout[i, j, k]
                pad_dx[h_start:h_start+fH, w_start:w_start+fW, :] += filters[k] * dout[i, j, k]
        dbiases[k] = np.sum(dout[:, :, k])

    # Remove padding from dx
    dx = pad_dx[padding:H_in+padding, padding:W_in+padding, :]

    return dx, dfilters, dbiases
