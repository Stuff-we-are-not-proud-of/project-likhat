import numpy as np
from ppm_to_matrix import ppm_to_matrix 

def generate_filters(filter_size, input_channels, num_filters=32):
    fan_in = filter_size[0] * filter_size[1] * input_channels
    limit = np.sqrt(6 / fan_in)
    filters = np.random.uniform(-limit, limit, size=(num_filters, filter_size[0], filter_size[1], input_channels))
    biases = np.zeros(num_filters)
    return filters, biases

def padding_image(image_matrix, pad):
    return np.pad(image_matrix, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

def single_convolution_layer(image_path, filter_size=(3, 3), stride=1, num_filters=32, padding=1):
    image_matrix = ppm_to_matrix(image_path)
    input_channels = image_matrix.shape[2]
    filters, biases = generate_filters(filter_size, input_channels, num_filters)
    padded_image = padding_image(image_matrix, pad=padding)
    input_height, input_width, _ = image_matrix.shape  # Unpadded size for output calculation
    output_height = ((input_height + 2*padding - filter_size[0]) // stride) + 1
    output_width = ((input_width + 2*padding - filter_size[1]) // stride) + 1
    output_matrix = np.zeros((output_height, output_width, num_filters))
    Z = np.zeros((output_height, output_width, num_filters))  # Pre-activation for cache
    
    for i in range(output_height):
        for j in range(output_width):
            subarray = padded_image[i*stride:i*stride + filter_size[0], j*stride:j*stride + filter_size[1], :]
            broadcasted_subarray = subarray[np.newaxis, :, :, :]
            multiplied = filters * broadcasted_subarray
            conv_sum = np.sum(multiplied, axis=(1,2,3))
            z = conv_sum + biases
            Z[i, j, :] = z
            output = np.maximum(0, z)
            output_matrix[i, j, :] = output
    
    cache = {
        "input": image_matrix,
        "padded_input": padded_image,
        "Z": Z,
        "A": output_matrix,
        "filters": filters,
        "biases": biases,
        "stride": stride,
        "padding": padding
    }
    
    return output_matrix, cache

def single_convolution_layer_from_matrix(image_matrix, filter_size=(3, 3), stride=1, num_filters=32, padding=1):
    input_channels = image_matrix.shape[2]
    filters, biases = generate_filters(filter_size, input_channels, num_filters)
    padded_image = padding_image(image_matrix, pad=padding)
    input_height, input_width, _ = image_matrix.shape  # Unpadded
    output_height = ((input_height + 2*padding - filter_size[0]) // stride) + 1
    output_width = ((input_width + 2*padding - filter_size[1]) // stride) + 1
    output_matrix = np.zeros((output_height, output_width, num_filters))
    Z = np.zeros((output_height, output_width, num_filters))  # Pre-activation for cache
    
    for i in range(output_height):
        for j in range(output_width):
            subarray = padded_image[i*stride:i*stride + filter_size[0], j*stride:j*stride + filter_size[1], :]
            broadcasted_subarray = subarray[np.newaxis, :, :, :]
            multiplied = filters * broadcasted_subarray
            conv_sum = np.sum(multiplied, axis=(1,2,3))
            z = conv_sum + biases
            Z[i, j, :] = z
            output = np.maximum(0, z)
            output_matrix[i, j, :] = output
    
    cache = {
        "input": image_matrix,
        "padded_input": padded_image,
        "Z": Z,
        "A": output_matrix,
        "filters": filters,
        "biases": biases,
        "stride": stride,
        "padding": padding
    }
    
    return output_matrix, cache

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