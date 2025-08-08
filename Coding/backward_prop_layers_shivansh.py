import numpy as np

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
