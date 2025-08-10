import numpy as np

def binary_crossentropy_loss(y_true, y_pred):
    return - (y_true * np.log(y_pred + 1e-10) + (1 - y_true) * np.log(1 - y_pred + 1e-10))

def binary_crossentropy_gradient(y_true, y_pred):
    return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-10)  

def dense_backward(dA, cache, activation):
    Z = cache['Z']
    A = cache['A']
    weights = cache['weights']
    input_data = cache['input']  
    
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    if Z.ndim == 1:
        Z = Z.reshape(1, -1)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if dA.ndim == 0 or dA.ndim == 1 and len(dA) == 1:
        dA = np.array([[dA.item() if dA.ndim == 1 else dA]]) 
    
    if activation == 'sigmoid':
        dZ = dA * A * (1 - A)
    elif activation == 'relu':
        dZ = dA * (Z > 0)
    else:
        dZ = dA
    
    dW = np.dot(input_data.T, dZ)
    db = np.sum(dZ, axis=0)
    dX = np.dot(dZ, weights.T)
    
    if dX.shape[0] == 1:
        dX = dX.flatten()
    
    return dX, dW, db

def flatten_backward(dA, cache):
    input_shape = cache['input_shape']
    dX = np.reshape(dA, input_shape)
    return dX

def dropout_backward(dA, cache):
    mask = cache['mask']
    dX = dA * mask
    return dX