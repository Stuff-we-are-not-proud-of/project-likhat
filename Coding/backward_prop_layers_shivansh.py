import numpy as np


def binary_crossentropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_gradient(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))
    
def dense_backward(dout, cache):
    x = cache["input"]
    W = cache["weights"]
    b = cache["biases"]
    z = cache["z"]
    activation = cache["activation"]

    # Activation derivative
    if activation == "ReLU":
        dz = dout * (z > 0)
    elif activation == "sigmoid":
        sig = 1 / (1 + np.exp(-z))
        dz = dout * sig * (1 - sig)
    else:  # Linear
        dz = dout

    # Gradients
    dW = np.outer(x, dz)
    db = dz
    dx = np.dot(W, dz)

    return dx, dW, db


def dropout_backward(dout, cache):

    if cache is None:
        return dout
    mask = cache["mask"]
    dx = dout * mask
    return dx


def flatten_backward(dout, cache):
    original_shape = cache["original_shape"]
    return dout.reshape(original_shape)
