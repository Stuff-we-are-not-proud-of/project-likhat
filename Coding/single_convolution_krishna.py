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
    
    for i in range(output_height):
        for j in range(output_width):
            subarray = padded_image[i*stride:i*stride + filter_size[0], j*stride:j*stride + filter_size[1], :]
            broadcasted_subarray = subarray[np.newaxis, :, :, :]
            multiplied = filters * broadcasted_subarray
            conv_sum = np.sum(multiplied, axis=(1,2,3))
            output = np.maximum(0, conv_sum + biases)
            output_matrix[i, j, :] = output
    
    return output_matrix 

def single_convolution_layer_from_matrix(image_matrix, filter_size=(3, 3), stride=1, num_filters=32, padding=1):
    input_channels = image_matrix.shape[2]
    filters, biases = generate_filters(filter_size, input_channels, num_filters)
    padded_image = padding_image(image_matrix, pad=padding)
    input_height, input_width, _ = image_matrix.shape  # Unpadded
    output_height = ((input_height + 2*padding - filter_size[0]) // stride) + 1
    output_width = ((input_width + 2*padding - filter_size[1]) // stride) + 1
    output_matrix = np.zeros((output_height, output_width, num_filters))
    
    for i in range(output_height):
        for j in range(output_width):
            subarray = padded_image[i*stride:i*stride + filter_size[0], j*stride:j*stride + filter_size[1], :]
            broadcasted_subarray = subarray[np.newaxis, :, :, :]
            multiplied = filters * broadcasted_subarray
            conv_sum = np.sum(multiplied, axis=(1,2,3))
            output = np.maximum(0, conv_sum + biases)
            output_matrix[i, j, :] = output
    
    return output_matrix 