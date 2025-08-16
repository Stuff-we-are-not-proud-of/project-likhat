import numpy as np
from ppm_to_matrix import ppm_to_matrix 
from numpy.lib.stride_tricks import as_strided

def generate_filters(filter_size, input_channels, num_filters=32):
    fan_in = filter_size[0] * filter_size[1] * input_channels
    limit = np.sqrt(6 / fan_in)
    filters = np.random.uniform(-limit, limit, size=(num_filters, filter_size[0], filter_size[1], input_channels))
    biases = np.zeros(num_filters)
    return filters, biases

def get_windows(input_matrix, kernel_size, stride=1, padding=0, dilate=0):
    h, w, c = input_matrix.shape
    if dilate > 0:
        input_matrix = np.insert(input_matrix, np.arange(1, h, 1), 0, axis=0)
        input_matrix = np.insert(input_matrix, np.arange(1, w, 1), 0, axis=1)
        h, w = input_matrix.shape[:2]
    input_matrix = np.pad(input_matrix, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    out_h = (h - kernel_size + 2 * padding) // stride + 1
    out_w = (w - kernel_size + 2 * padding) // stride + 1
    h_str, w_str, c_str = input_matrix.strides
    windows = as_strided(input_matrix, (out_h, out_w, kernel_size, kernel_size, c), (stride * h_str, stride * w_str, h_str, w_str, c_str))
    return windows

def single_convolution_layer(image_path, filter_size=(3, 3), stride=1, num_filters=32, padding=1, filters=None, biases=None):
    image_matrix = ppm_to_matrix(image_path)
    input_channels = image_matrix.shape[2]
    if filters is None or biases is None:
        filters, biases = generate_filters(filter_size, input_channels, num_filters)
    input_height, input_width, _ = image_matrix.shape
    output_height = ((input_height + 2*padding - filter_size[0]) // stride) + 1
    output_width = ((input_width + 2*padding - filter_size[1]) // stride) + 1
    windows = get_windows(image_matrix, filter_size[0], stride, padding)
    Z = np.einsum('ijabc,oabc->oij', windows, filters) + biases[None, None, :]
    Z = Z.transpose(1, 2, 0)
    output_matrix = np.maximum(0, Z)
    cache = {
        "input": image_matrix,
        "windows": windows,
        "Z": Z,
        "A": output_matrix,
        "filters": filters,
        "biases": biases,
        "stride": stride,
        "padding": padding
    }
    return output_matrix, cache

def single_convolution_layer_from_matrix(image_matrix, filter_size=(3, 3), stride=1, num_filters=32, padding=1, filters=None, biases=None):
    input_channels = image_matrix.shape[2]
    if filters is None or biases is None:
        filters, biases = generate_filters(filter_size, input_channels, num_filters)
    input_height, input_width, _ = image_matrix.shape
    output_height = ((input_height + 2*padding - filter_size[0]) // stride) + 1
    output_width = ((input_width + 2*padding - filter_size[1]) // stride) + 1
    windows = get_windows(image_matrix, filter_size[0], stride, padding)
    Z = np.einsum('ijabc,oabc->oij', windows, filters) + biases[None, None, :]
    Z = Z.transpose(1, 2, 0)
    output_matrix = np.maximum(0, Z)
    cache = {
        "input": image_matrix,
        "windows": windows,
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
    windows = cache['windows']
    Z = cache['Z']
    filters = cache['filters']
    biases = cache['biases']
    stride = cache['stride']
    padding = cache['padding']
    
    dZ = dA * (Z > 0)
    
    num_filters, f_h, f_w, f_c = filters.shape
    input_height, input_width, input_channels = input_data.shape
    output_height, output_width, _ = dZ.shape
    
    db = np.sum(dZ, axis=(0,1))
    dw = np.einsum('kij,ijlmn->klmn', dZ.transpose(2,0,1), windows)
    
    rot_filters = np.flip(filters, (1,2))
    back_padding = f_h - 1
    dilate = stride - 1 if stride > 1 else 0
    dout_windows = get_windows(dZ, f_h, stride=1, padding=back_padding, dilate=dilate)
    dX = np.einsum('ijabk,kabc->ijc', dout_windows, rot_filters)
    
    return dX, dw, db