import numpy as np

def conv_backward(dA, cache):
    input_data = cache['input']
    padded_input = cache['padded_input']
    Z = cache['Z']
    filters = cache['filters']
    biases = cache['biases']
    stride = cache['stride']
    padding = cache['padding']
    
    dZ = dA * (Z > 0)
    
    num_filters, f_h, f_w, f_c = filters.shape
    input_height, input_width, input_channels = input_data.shape
    output_height, output_width, _ = dZ.shape
    
    dW = np.zeros_like(filters, dtype=np.float64)
    db = np.sum(dZ, axis=(0,1))
    dX_padded = np.zeros_like(padded_input, dtype=np.float64)
    
    for i in range(output_height):
        for j in range(output_width):
            for k in range(num_filters):
                sub_input = padded_input[i*stride:i*stride + f_h, j*stride:j*stride + f_w, :]
                dW[k] += sub_input * dZ[i, j, k]
                dX_padded[i*stride:i*stride + f_h, j*stride:j*stride + f_w, :] += filters[k] * dZ[i, j, k]
    
    dX = dX_padded[padding:-padding, padding:-padding, :]
    
    return dX, dW, db