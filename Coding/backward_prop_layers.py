import numpy as np

def dense_backward(dA, cache, activation):
    Z = cache['Z']
    A = cache['A']
    weights = cache['weights']
    input = cache['input']
    
    if activation == 'sigmoid':
        dZ = dA * A * (1 - A)
    elif activation == 'relu':
        dZ = dA * (Z > 0)
    else:
        dZ = dA
    
    dW = np.dot(input.T, dZ)
    db = np.sum(dZ, axis=0)
    dX = np.dot(dZ, weights.T)
    
    return dX, dW, db

def flatten_backward(dA, cache):
    input_shape = cache['input_shape']
    dX = np.reshape(dA, input_shape)
    return dX

def dropout_backward(dA, cache):
    mask = cache['mask']
    dX = dA * mask
    return dX