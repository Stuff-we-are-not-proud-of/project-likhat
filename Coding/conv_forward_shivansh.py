import numpy as np
from ppm_to_matrix_shivansh import extract_rgb_matrix

def initialize_conv_layer(num_filters, filter_size, in_channels):
    """
    He initialization for convolutional filters.
    """
    filters = np.random.randn(num_filters, filter_size, filter_size, in_channels) * np.sqrt(2. / (filter_size * filter_size * in_channels))
    biases = np.zeros(num_filters)
    return filters, biases

def pad(X, padding):
    """
    Zero padding for input matrix.
    """
    if padding > 0:
        return np.pad(X, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)
    return X

def conv_forward(image_path, num_filters, filter_size, stride=1, padding=1):
    """
    Convolution forward pass from an image path.
    Always initializes filters/biases inside.
    """
    input_matrix = extract_rgb_matrix(image_path)
    return conv_forward_from_matrix(input_matrix, num_filters, filter_size, stride, padding)

def conv_forward_from_matrix(input_matrix, num_filters, filter_size, stride=1, padding=0):
    """
    Convolution forward pass from a NumPy matrix.
    Always initializes filters/biases inside.
    """
    H_in, W_in, C_in = input_matrix.shape

    # Always init inside
    filters, biases = initialize_conv_layer(num_filters, filter_size, C_in)

    # Pad input
    padded_input = pad(input_matrix, padding)

    # Output dimensions
    H_out = (H_in + 2*padding - filter_size) // stride + 1
    W_out = (W_in + 2*padding - filter_size) // stride + 1
    Z = np.zeros((H_out, W_out, num_filters))

    # Convolution loop
    for f in range(num_filters):
        filt = filters[f]
        bias = biases[f]
        for i in range(H_out):
            for j in range(W_out):
                region = padded_input[i*stride:i*stride+filter_size, j*stride:j*stride+filter_size, :]
                if region.shape == filt.shape:
                    Z[i, j, f] = np.sum(region * filt) + bias


    Z = np.maximum(0, Z)

    # Cache for backprop
    cache = {
        "input": input_matrix,
        "filters": filters,
        "biases": biases,
        "stride": stride,
        "padding": padding,

        "output_shape": Z.shape
    }
    return Z, cache
