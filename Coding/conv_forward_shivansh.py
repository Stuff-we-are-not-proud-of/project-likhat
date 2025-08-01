import numpy as np
from PIL import Image
from ppm_to_matrix_shivansh import extract_rgb_matrix


def pad(X):

    X_pad = np.pad(X,((1,1),(1,1),(0,0)),mode='constant',constant_values=(0,0))
    
    return X_pad

def initialize_conv_layer(num_filters, filter_size, in_channels):
    filters = np.random.randn(num_filters, filter_size, filter_size, in_channels) * np.sqrt(2. / (filter_size * filter_size * in_channels))   #He initialization
    biases = np.zeros(num_filters)
    return filters, biases

def conv_forward(image_path, filters, biases, stride=1,padding=1):
    """
    input_image: np.array of shape (H_in, W_in, C_in)
    filters: np.array of shape (num_filters, filter_height, filter_width, C_in)
    biases: np.array of shape (num_filters,)
    stride: int, default 1

    Returns: output of shape (H_out, W_out, num_filters)
    """
    
    rgb_matrix=extract_rgb_matrix(image_path)
    H_in, W_in, C_in = rgb_matrix.shape
    num_filters, F_h, F_w, _ = filters.shape

    # Output dimensions
    H_out = (H_in + 2*padding - F_h) // stride + 1
    W_out = (W_in + 2*padding  - F_w) // stride + 1
    Z = np.zeros((H_out, W_out, num_filters))
    pad(rgb_matrix)
    # Perform convolution
    for f in range(num_filters):
        filt = filters[f]
        bias = biases[f]
        for i in range(0, H_out):
            for j in range(0, W_out):
                region = rgb_matrix[i*stride:i*stride+F_h, j*stride:j*stride+F_w, :]   #slicing the layers 
                if region.shape==filt.shape:                                           #checking dimensions of filter and the slice which was giving the error
                    
                    Z[i, j, f] = np.sum(region * filt) + bias
                else:
                    pass 
    Z=np.maximum(0,Z)   
    return Z

